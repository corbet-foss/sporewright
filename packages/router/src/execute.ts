// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Fallback chain execution — the LLM orchestrator over `runAttemptChain`.
 *
 * `callWithChain` tries each provider in the resolved chain sequentially. On any
 * error (429, timeout, bad output), it moves to the next provider. An optional
 * critic can reject results that parsed but are low quality. Returns the result,
 * the provider that succeeded, and token usage.
 *
 * Three host-injected seams keep this bundler-agnostic and side-effect-free:
 *   - `onTrace`  — the host wires telemetry (e.g. a gateway sink) here; the
 *                  engine emits every AttemptTrace and writes nothing itself.
 *   - `logger`   — an injected `{ log, warn }` (default no-op); the router carries
 *                  no `import.meta.env` / build-tool assumption.
 *   - `state`    — the cooldown / dedup state (default module singleton), so
 *                  process-scoped behaviour and the test-clear seam are preserved.
 */

import type { LanguageModel } from 'ai';
import { planExploration } from 'sporewright';
import { runAttemptChain, type AttemptTrace } from './attempt-chain';
import { learningEventForTrace } from './learning';
import { routingPlanOf } from './resolve';
import {
    type RouterState,
    defaultRouterState,
    activeCooldownSeconds,
    setProviderCooldown,
    shouldSuppressRepeat,
} from './cooldown';
import {
    describeError,
    isRateLimitError,
    isCorsError,
    isProviderLevelFailure,
    extractRetryAfter,
} from './classify';
import { getAiModule, createModel } from './model-factory';
import {
    type ProviderSlot,
    type ProviderFailure,
    type ChainAttempt,
    type UsageInfo,
    type CallResult,
    type RoutingExplanation,
    type RouterLogger,
    type CallWithChainOptions,
    RateLimitError,
    ChainExhaustionError,
} from './errors';

/** No-op logger — the default when the host injects none. */
const noopLogger: RouterLogger = { log() {}, warn() {} };

/**
 * Execute an LLM call with provider fallback. Tries each provider in the
 * chain sequentially. On any error (429, timeout, bad output), moves to
 * the next provider. An optional critic function can reject results that
 * parsed successfully but are low quality (e.g., missing critical fields).
 *
 * Returns the result, the ID of the provider that succeeded, and token usage.
 */
export async function callWithChain<T>(
    chain: ProviderSlot[],
    callFn: (model: LanguageModel, signal: AbortSignal) => Promise<{ result: T; usage?: UsageInfo }>,
    options?: CallWithChainOptions<T>,
): Promise<CallResult<T>> {
    if (chain.length === 0) {
        throw new Error('No LLM providers configured. Add an API key in Cloud Services.');
    }

    const aiMod = await getAiModule();
    const label = options?.label || 'call';
    const log = options?.logger ?? noopLogger;
    const state: RouterState = options?.state ?? defaultRouterState;
    const routingPlan = routingPlanOf(chain);
    const modelFactory = options?.modelFactory ?? (async (slot: ProviderSlot) => {
        if (slot.apiKey === undefined) {
            throw new Error(`Provider "${slot.providerId}" requires a host execution adapter.`);
        }
        return createModel(slot.providerId, slot.apiKey, slot.model);
    });

    const seen = new Set<string>();
    const deduped = chain.filter(s => {
        const key = `${s.providerId}|${s.model}|${s.apiKey ?? 'credentialless'}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    const chainIds = deduped.map(s => s.providerId);
    log.log(`${label}: chain=[${chainIds.join(', ')}]${deduped.length < chain.length ? ` (${chain.length - deduped.length} duplicates removed)` : ''}`);

    const failures: ProviderFailure[] = [];
    const attempts: ChainAttempt[] = [];
    let lastRetryAfter: number | undefined;
    const skipProviderModel = new Set<string>();

    const emitAttempt = async (trace: AttemptTrace<ProviderSlot>) => {
        const attempt = chainAttemptFromTrace(trace);
        attempts.push(attempt);

        if (trace.status === 'failed') {
            failures.push({
                providerId: trace.attempt.providerId,
                model: trace.attempt.model,
                error: trace.error ?? 'provider failed',
                statusCode: trace.statusCode,
            });
            if (trace.statusCode === 429 || trace.retryAfterSeconds !== undefined) {
                lastRetryAfter = trace.retryAfterSeconds ?? lastRetryAfter ?? 300;
            }
        } else if (trace.status === 'rejected') {
            failures.push({
                providerId: trace.attempt.providerId,
                model: trace.attempt.model,
                error: trace.error ?? 'critic rejected',
            });
        }

        // The host wires telemetry (e.g. a gateway sink) via onTrace; the engine
        // writes nothing itself.
        try {
            options?.onTrace?.(trace);
        } catch (err) {
            console.warn(`[callWithChain] trace observer failed for ${label}`, err);
        }
        logTrace(trace, deduped, skipProviderModel, label, log, state);

        try {
            await options?.onAttempt?.(attempt);
        } catch (err) {
            console.warn(`[callWithChain] attempt observer failed for ${label}`, err);
        }
        const learningEvent = learningEventForTrace(routingPlan, trace);
        if (learningEvent !== undefined) {
            try {
                await options?.onLearningEvent?.(learningEvent);
            } catch (err) {
                console.warn(`[callWithChain] learning observer failed for ${label}`, err);
            }
        }
    };

    const run = await runAttemptChain<ProviderSlot, undefined, { result: T; usage?: UsageInfo }>(
        deduped,
        undefined,
        async ({ attempt, signal }) => {
            const model = await modelFactory(attempt);
            return callFn(model, signal);
        },
        {
            signal: options?.signal,
            attemptTimeoutMs: options?.attemptTimeoutMs,
            continueOnLastCriticReject: true,
            shouldSkip: (slot) => {
                const providerModelKey = `${slot.providerId}|${slot.model}`;
                if (skipProviderModel.has(providerModelKey)) {
                    return { status: 'silent', reason: 'provider-model skipped' };
                }

                const cooldownSeconds = activeCooldownSeconds(state, slot.providerId, slot.model);
                if (cooldownSeconds !== undefined) {
                    skipProviderModel.add(providerModelKey);
                    lastRetryAfter = cooldownSeconds;
                    return {
                        status: 'failed',
                        reason: `Provider cooldown active; retry after ${cooldownSeconds}s`,
                        statusCode: 429,
                        retryAfterSeconds: cooldownSeconds,
                        metadata: { executed: false },
                    };
                }

                return false;
            },
            critic: options?.critic ? (callResult) => options.critic!(callResult.result) : undefined,
            describeResult: (callResult) => ({
                ...(callResult.usage ? { usage: callResult.usage } : {}),
                ...qualityLossMetadata(options?.qualityLoss, callResult.result),
            }),
            classifyError: (error, slot) => {
                const desc = describeError(error, aiMod);
                const rateLimited = isRateLimitError(error, aiMod);
                const retryAfter = rateLimited ? extractRetryAfter(error, aiMod) : undefined;
                const providerLevelFailure = isProviderLevelFailure(error, aiMod);
                if (rateLimited) {
                    setProviderCooldown(state, slot.providerId, slot.model, retryAfter);
                    lastRetryAfter = retryAfter ?? lastRetryAfter ?? 300;
                }
                if (providerLevelFailure) {
                    skipProviderModel.add(`${slot.providerId}|${slot.model}`);
                }
                return {
                    message: desc.message,
                    statusCode: desc.statusCode,
                    retryAfterSeconds: retryAfter,
                    providerLevelFailure,
                    metadata: {
                        rateLimited,
                        cors: isCorsError(error, aiMod),
                    },
                };
            },
            onTrace: emitAttempt,
        },
    );

    if (run.ok) {
        await runCuriosityProbes(
            deduped,
            run.trace,
            callFn,
            options,
            routingPlan,
            aiMod,
            state,
            modelFactory,
        );
        return {
            result: run.result.result,
            providerId: run.attempt.providerId,
            model: run.attempt.model,
            wasFallback: run.trace.find(trace => trace.ok && trace.attempt === run.attempt)?.wasFallback ?? false,
            usage: run.result.usage,
            attempts,
            routing: routingExplanation(routingPlan, run.attempt, attempts),
        };
    }

    if (failures.length > 0 && failures.every(f => f.statusCode === 429 || isRateLimitError(new Error(f.error), aiMod))) {
        throw new RateLimitError(failures.map(f => f.providerId), lastRetryAfter, attempts);
    }

    throw new ChainExhaustionError(failures, attempts);
}

async function runCuriosityProbes<T>(
    chain: ProviderSlot[],
    primaryTrace: AttemptTrace<ProviderSlot>[],
    callFn: (model: LanguageModel, signal: AbortSignal) => Promise<{ result: T; usage?: UsageInfo }>,
    options: CallWithChainOptions<T> | undefined,
    routingPlan: ReturnType<typeof routingPlanOf>,
    aiMod: Awaited<ReturnType<typeof getAiModule>>,
    state: RouterState,
    modelFactory: (slot: ProviderSlot) => Promise<LanguageModel>,
): Promise<void> {
    const policy = routingPlan?.exploration;
    const budget = options?.explorationBudget;
    if (
        routingPlan === undefined
        || policy === undefined
        || budget === undefined
        || routingPlan.decision.policy.temperature <= 0
        || policy.maxExecutions <= 1
        || options?.signal?.aborted
    ) return;

    const firstSlot = new Map<string, ProviderSlot>();
    for (const slot of chain) {
        const option = `${slot.providerId}\0${slot.model}`;
        if (!firstSlot.has(option)) firstSlot.set(option, slot);
    }
    const expectedCosts = Object.fromEntries([...firstSlot].map(([option]) => [option, 1]));
    const planned = planExploration(routingPlan.decision, expectedCosts, {
        max_executions: Math.max(1, Math.min(3, policy.maxExecutions)),
        max_total_cost: Math.max(1, Math.min(3, policy.maxExecutions)),
    });
    if (typeof planned === 'string') return;

    const alreadyObserved = new Set(primaryTrace
        .filter(({ status }) => status !== 'skipped')
        .map(({ attempt }) => `${attempt.providerId}\0${attempt.model}`));
    const probes = planned.executions
        .filter(({ option }) => !alreadyObserved.has(option))
        .slice(0, Math.max(0, policy.maxExecutions - 1));

    await Promise.all(probes.map(async ({ option }) => {
        const slot = firstSlot.get(option);
        if (slot === undefined || options?.signal?.aborted) return;
        const reservation = await budget.reserve({
            workspaceId: routingPlan.workspaceId,
            accountId: policy.accountId,
            fundingKind: policy.fundingKind,
            estimatedUnits: policy.reservationUnits,
            providerId: slot.providerId,
            model: slot.model,
        }).catch(() => undefined);
        if (reservation === undefined) return;

        let actualUnits = reservation.reservedUnits;
        try {
            const probe = await runAttemptChain<ProviderSlot, undefined, { result: T; usage?: UsageInfo }>(
                [slot],
                undefined,
                async ({ attempt, signal }) => {
                    const model = await modelFactory(attempt);
                    return callFn(model, signal);
                },
                {
                    signal: options?.signal,
                    attemptTimeoutMs: options?.attemptTimeoutMs,
                    continueOnLastCriticReject: true,
                    critic: options?.critic ? (callResult) => options.critic!(callResult.result) : undefined,
                    describeResult: (callResult) => ({
                        ...(callResult.usage ? { usage: callResult.usage } : {}),
                        ...qualityLossMetadata(options?.qualityLoss, callResult.result),
                        exploratory: true,
                    }),
                    classifyError: (error, attempted) => {
                        const desc = describeError(error, aiMod);
                        const rateLimited = isRateLimitError(error, aiMod);
                        const retryAfter = rateLimited ? extractRetryAfter(error, aiMod) : undefined;
                        if (rateLimited) {
                            setProviderCooldown(state, attempted.providerId, attempted.model, retryAfter);
                        }
                        return {
                            message: desc.message,
                            statusCode: desc.statusCode,
                            retryAfterSeconds: retryAfter,
                            providerLevelFailure: isProviderLevelFailure(error, aiMod),
                            metadata: {
                                exploratory: true,
                                rateLimited,
                                cors: isCorsError(error, aiMod),
                            },
                        };
                    },
                    onTrace: async (trace) => {
                        try {
                            options?.onTrace?.(trace);
                        } catch (err) {
                            console.warn(`[callWithChain] curiosity trace observer failed for ${options?.label ?? 'call'}`, err);
                        }
                        const learningEvent = learningEventForTrace(routingPlan, trace);
                        if (learningEvent !== undefined) {
                            try {
                                await options?.onLearningEvent?.(learningEvent);
                            } catch (err) {
                                console.warn(`[callWithChain] curiosity learning observer failed for ${options?.label ?? 'call'}`, err);
                            }
                        }
                    },
                },
            );
            if (probe.ok && probe.result.usage?.totalTokens !== undefined) {
                actualUnits = probe.result.usage.totalTokens;
            }
        } catch {
            // Exploration is best-effort after the primary result is already safe.
        } finally {
            await budget.settle({
                reservationId: reservation.reservationId,
                actualUnits,
            }).catch(() => {});
        }
    }));
}

function routingExplanation(
    plan: ReturnType<typeof routingPlanOf>,
    selected: ProviderSlot,
    attempts: readonly ChainAttempt[],
): RoutingExplanation {
    const recommended = plan?.recommended[0] ?? {
        provider: selected.providerId,
        model: selected.model,
    };
    const failed = attempts.find((attempt) => attempt.status !== 'selected');
    let code: RoutingExplanation['code'] = plan?.policyMode === 'fixed'
        ? 'fixed_by_user'
        : plan?.policyMode === 'prefer'
            ? 'preferred_by_user'
            : 'smart_choice';
    const selectedRecommended = recommended.provider === selected.providerId
        && recommended.model === selected.model;
    if (failed === undefined && plan?.policyMode === 'auto' && !selectedRecommended) {
        // Resolution can remove a learned recommendation before execution when
        // credentials, device capabilities, or host policy make it unavailable.
        // That absence creates no attempt trace, but it is still the truthful
        // user-facing reason for choosing another route.
        code = 'provider_unavailable';
    } else if (failed !== undefined) {
        if (failed.statusCode === 429) code = 'rate_limited';
        else if (failed.statusCode === 402) code = 'credits_unavailable';
        else if (failed.statusCode === 401 || failed.statusCode === 403) code = 'credentials_rejected';
        else if (failed.status === 'rejected' || failed.criticRejected) code = 'quality_rejected';
        else code = 'provider_unavailable';
    }
    return {
        code,
        policyMode: plan?.policyMode ?? 'auto',
        recommendedProviderId: recommended.provider,
        recommendedModel: recommended.model,
        selectedProviderId: selected.providerId,
        selectedModel: selected.model,
    };
}

function boundedQualityLoss(value: number): number {
    return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 1;
}

function qualityLossMetadata<T>(
    measure: ((result: T) => number) | undefined,
    result: T,
): { qualityLoss?: number } {
    if (measure === undefined) return {};
    try {
        return { qualityLoss: boundedQualityLoss(measure(result)) };
    } catch {
        return {};
    }
}

function traceUsage(trace: AttemptTrace<ProviderSlot>): UsageInfo | undefined {
    const usage = trace.metadata?.usage;
    if (!usage || typeof usage !== 'object') return undefined;
    const candidate = usage as Partial<UsageInfo>;
    if (
        typeof candidate.inputTokens === 'number' &&
        typeof candidate.outputTokens === 'number' &&
        typeof candidate.totalTokens === 'number'
    ) {
        return {
            inputTokens: candidate.inputTokens,
            outputTokens: candidate.outputTokens,
            totalTokens: candidate.totalTokens,
        };
    }
    return undefined;
}

function chainAttemptFromTrace(trace: AttemptTrace<ProviderSlot>): ChainAttempt {
    return {
        attemptNo: trace.attemptNo,
        providerId: trace.attempt.providerId,
        model: trace.attempt.model,
        status: trace.status === 'skipped' ? 'failed' : trace.status,
        latencyMs: trace.latencyMs,
        wasFallback: trace.wasFallback,
        ...(traceUsage(trace) ? { usage: traceUsage(trace) } : {}),
        ...(trace.error ? { error: trace.error } : {}),
        ...(trace.statusCode !== undefined ? { statusCode: trace.statusCode } : {}),
        ...(trace.retryAfterSeconds !== undefined ? { retryAfterSeconds: trace.retryAfterSeconds } : {}),
        ...(trace.criticRejected !== undefined ? { criticRejected: trace.criticRejected } : {}),
    };
}

/**
 * Default trace-logging observer, driven by the injected logger. The host can
 * replace this entirely via its own `onTrace`/`logger`; with the default no-op
 * logger it produces no output (no build-tool assumption).
 */
function logTrace(
    trace: AttemptTrace<ProviderSlot>,
    deduped: ProviderSlot[],
    skipProviderModel: Set<string>,
    label: string,
    log: RouterLogger,
    state: RouterState,
): void {
    if (trace.status === 'rejected') {
        log.warn(`${label}: critic rejected ${trace.attempt.providerId}:${trace.attempt.model}, trying next`);
        return;
    }

    if (trace.status === 'selected' && trace.criticRejected) {
        log.warn(`${label}: critic rejected ${trace.attempt.providerId}:${trace.attempt.model} (last provider, returning degraded result)`);
        return;
    }

    if (trace.status !== 'failed' && trace.status !== 'skipped') {
        return;
    }

    const slot = trace.attempt;
    const providerModelKey = `${slot.providerId}|${slot.model}`;
    if (trace.providerLevelFailure) {
        const skipped = deduped.slice(deduped.indexOf(slot) + 1).filter(s => `${s.providerId}|${s.model}` === providerModelKey).length;
        if (skipped > 0) log.warn(`${label}: ${slot.providerId} provider-level failure, skipping ${skipped} remaining key(s)`);
    }

    const index = deduped.indexOf(slot);
    const nextSlot = deduped.slice(index + 1).find(s => !skipProviderModel.has(`${s.providerId}|${s.model}`));
    const rateLimited = trace.statusCode === 429 || Boolean(trace.metadata?.rateLimited);
    const retryAfter = trace.retryAfterSeconds;
    if (rateLimited) {
        const dedupKey = `429:${slot.providerId}`;
        if (!shouldSuppressRepeat(state, dedupKey)) {
            log.warn(`${label}: ${slot.providerId} rate-limited (HTTP ${trace.statusCode || 429})${retryAfter ? `, retry after ${retryAfter}s` : ''}${nextSlot ? `, trying ${nextSlot.providerId}` : ''}`);
        }
        return;
    }

    if (Boolean(trace.metadata?.cors)) {
        const dedupKey = `cors:${slot.providerId}`;
        if (!shouldSuppressRepeat(state, dedupKey)) {
            log.warn(`${label}: ${slot.providerId} CORS block (direct browser->API calls not allowed from this origin). Provider skipped.${nextSlot ? ` Falling back to ${nextSlot.providerId}.` : ''}`);
        }
        return;
    }

    const code = trace.statusCode ? `HTTP ${trace.statusCode}` : (trace.error ?? 'provider failed').slice(0, 60);
    log.warn(`${label}: ${slot.providerId} failed (${code})${nextSlot ? `, trying ${nextSlot.providerId}` : ''}`);
}
