// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Generic provider-fallback execution engine.
 *
 * This module is intentionally NOT concerned with how the chain was resolved.
 * The cascade tensor (`cascade-tensor.ts` / `resolveChain`) decides WHICH
 * (provider, model) options to try and in what order. This module is only the
 * loop that executes over that already-resolved ordered list: try each attempt,
 * classify its outcome, fall through on failure/rejection, and emit a trace per
 * attempt.
 *
 * `callWithChain` (`execute.ts`) is the only consumer; its call site is unchanged
 * except for the import path.
 */

// ---------------------------------------------------------------------------
// Public types — the `AttemptTrace` family.
// ---------------------------------------------------------------------------

/** Outcome class for a single attempt in the chain. */
export type AttemptStatus = 'selected' | 'rejected' | 'failed' | 'skipped';

/**
 * Classification of a thrown execution error, produced by the caller's
 * `classifyError`. Drives status-code/retry/metadata fields on the trace and
 * lets the caller mark a failure as provider-level (so siblings sharing the
 * same provider+model can be short-circuited upstream).
 */
export interface AttemptFailure {
    message: string;
    statusCode?: number;
    retryAfterSeconds?: number;
    providerLevelFailure?: boolean;
    metadata?: Record<string, unknown>;
}

/**
 * Result of a pre-attempt `shouldSkip` check.
 * - `status: 'silent'` drops the attempt with no trace entry (e.g. a sibling key
 *   whose provider+model was already skipped).
 * - `status: 'failed'` records the skip as a failed trace entry (e.g. an active
 *   cooldown that should surface as a 429-style failure).
 * - `status: 'skipped'` (default) records it as a skipped trace entry.
 */
export interface AttemptSkip {
    reason: string;
    status?: 'failed' | 'skipped' | 'silent';
    statusCode?: number;
    retryAfterSeconds?: number;
    providerLevelFailure?: boolean;
    metadata?: Record<string, unknown>;
}

/** One emitted trace entry, generic over the attempt slot type `A`. */
export interface AttemptTrace<A> {
    attempt: A;
    attemptNo: number;
    status: AttemptStatus;
    ok: boolean;
    latencyMs: number;
    wasFallback: boolean;
    error?: string;
    statusCode?: number;
    retryAfterSeconds?: number;
    criticRejected?: boolean;
    providerLevelFailure?: boolean;
    metadata?: Record<string, unknown>;
}

/** Read-only snapshot of chain progress passed to `shouldSkip`. */
export interface AttemptChainState<A> {
    index: number;
    attemptNo: number;
    trace: AttemptTrace<A>[];
}

/** Per-attempt execution context handed to the `execute` callback. */
export interface AttemptExecution<A, I> {
    input: I;
    attempt: A;
    attemptNo: number;
    signal: AbortSignal;
}

/**
 * Critic predicate. Returning `true` accepts the result; returning `false` or a
 * string rejects it (the string becomes the rejection reason). Async supported.
 */
export type AttemptCritic<O, A> =
    | ((result: O, attempt: A) => Promise<boolean | string> | boolean | string)
    | undefined;

export interface AttemptChainOptions<A, O> {
    signal?: AbortSignal | undefined;
    attemptTimeoutMs?: number | undefined;
    critic?: AttemptCritic<O, A>;
    /**
     * When the LAST attempt's result is critic-rejected, return it anyway as a
     * degraded `selected` result (ok=true, criticRejected=true) instead of
     * failing the whole chain.
     */
    continueOnLastCriticReject?: boolean;
    shouldSkip?: (
        attempt: A,
        state: AttemptChainState<A>,
    ) => Promise<AttemptSkip | string | false | null | undefined> | AttemptSkip | string | false | null | undefined;
    classifyError?: (error: unknown, attempt: A) => AttemptFailure;
    describeResult?: (result: O, attempt: A) => Record<string, unknown> | undefined;
    onTrace?: (trace: AttemptTrace<A>) => Promise<void> | void;
    /** Injectable clock for deterministic latency tests. */
    now?: () => number;
}

export type AttemptChainResult<A, O> =
    | { ok: true; result: O; trace: AttemptTrace<A>[]; attempt: A }
    | { ok: false; trace: AttemptTrace<A>[]; error: string };

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/**
 * Execute an ordered list of `attempts` one at a time, falling through on
 * failure or critic rejection, until one is accepted or the list is exhausted.
 *
 * Semantics:
 *
 * - A caller-fired `signal.aborted` (before an attempt, or surfacing as the
 *   thrown error during one) THROWS the abort reason — it never consumes a
 *   fallback attempt. This keeps "user clicked ABORT / closed the tab" distinct
 *   from "provider failed".
 * - `shouldSkip` runs before each attempt. `'silent'` skips with no trace;
 *   otherwise a skip becomes a trace entry with the given status/metadata and
 *   the chain continues.
 * - Each execution is wrapped by `withAttemptTimeout`: a per-attempt timeout (or
 *   the parent signal) aborts the call; a timeout surfaces as a thrown
 *   TimeoutError that the catch block records as a `failed` attempt.
 * - On success, the `critic` (if any) decides acceptance. Rejection on a
 *   non-last attempt records a `rejected` trace and continues; rejection on the
 *   last attempt with `continueOnLastCriticReject` returns the degraded result
 *   as `selected` (ok=true, criticRejected=true).
 * - `describeResult` attaches caller metadata (e.g. token usage) to selected
 *   and last-critic-rejected traces.
 * - On a thrown error, `classifyError` normalizes it into status code / retry /
 *   provider-level / metadata fields recorded on the `failed` trace.
 * - `wasFallback` is `index > 0` for every trace entry.
 * - `onTrace` is awaited after each emitted entry.
 * - If no attempt is accepted, returns `{ ok: false, error: 'no route attempt succeeded' }`.
 */
export async function runAttemptChain<A, I, O>(
    attempts: readonly A[],
    input: I,
    execute: (execution: AttemptExecution<A, I>) => Promise<O> | O,
    options: AttemptChainOptions<A, O> = {},
): Promise<AttemptChainResult<A, O>> {
    const trace: AttemptTrace<A>[] = [];
    const now = options.now ?? (() => Date.now());

    for (let index = 0; index < attempts.length; index++) {
        if (options.signal?.aborted) {
            throw abortError(options.signal.reason);
        }

        const attempt = attempts[index]!;
        const attemptNo = trace.length + 1;
        const state = { index, attemptNo, trace };
        const skip = await options.shouldSkip?.(attempt, state);
        if (skip) {
            const normalized = normalizeSkip(skip);
            if (normalized.status === 'silent') {
                continue;
            }
            const skipped: AttemptTrace<A> = {
                attempt,
                attemptNo,
                status: normalized.status ?? 'skipped',
                ok: false,
                latencyMs: 0,
                wasFallback: index > 0,
                error: normalized.reason,
                ...(normalized.statusCode !== undefined ? { statusCode: normalized.statusCode } : {}),
                ...(normalized.retryAfterSeconds !== undefined ? { retryAfterSeconds: normalized.retryAfterSeconds } : {}),
                ...(normalized.providerLevelFailure !== undefined ? { providerLevelFailure: normalized.providerLevelFailure } : {}),
                ...(normalized.metadata ? { metadata: normalized.metadata } : {}),
            };
            trace.push(skipped);
            await options.onTrace?.(skipped);
            continue;
        }

        const start = now();
        try {
            const result = await withAttemptTimeout(
                (signal) => execute({ input, attempt, attemptNo, signal }),
                options.signal,
                options.attemptTimeoutMs,
            );
            const critic = options.critic ? await options.critic(result, attempt) : true;
            const isLast = index === attempts.length - 1;
            if (critic !== true) {
                const reason = typeof critic === 'string' ? critic : 'critic rejected result';
                const metadata = options.describeResult?.(result, attempt);
                const rejected: AttemptTrace<A> = {
                    attempt,
                    attemptNo,
                    status: isLast && options.continueOnLastCriticReject ? 'selected' : 'rejected',
                    ok: Boolean(isLast && options.continueOnLastCriticReject),
                    latencyMs: Math.max(0, now() - start),
                    wasFallback: index > 0,
                    error: reason,
                    criticRejected: true,
                    ...(metadata ? { metadata } : {}),
                };
                trace.push(rejected);
                await options.onTrace?.(rejected);
                if (isLast && options.continueOnLastCriticReject) {
                    return { ok: true, result, trace, attempt };
                }
                continue;
            }

            const selected: AttemptTrace<A> = {
                attempt,
                attemptNo,
                status: 'selected',
                ok: true,
                latencyMs: Math.max(0, now() - start),
                wasFallback: index > 0,
                ...(() => {
                    const metadata = options.describeResult?.(result, attempt);
                    return metadata ? { metadata } : {};
                })(),
            };
            trace.push(selected);
            await options.onTrace?.(selected);
            return { ok: true, result, trace, attempt };
        } catch (error) {
            if (options.signal?.aborted) {
                throw abortError(options.signal.reason);
            }
            const failure = options.classifyError?.(error, attempt) ?? defaultFailure(error);
            const failed: AttemptTrace<A> = {
                attempt,
                attemptNo,
                status: 'failed',
                ok: false,
                latencyMs: Math.max(0, now() - start),
                wasFallback: index > 0,
                error: failure.message,
                ...(failure.statusCode !== undefined ? { statusCode: failure.statusCode } : {}),
                ...(failure.retryAfterSeconds !== undefined ? { retryAfterSeconds: failure.retryAfterSeconds } : {}),
                ...(failure.providerLevelFailure !== undefined ? { providerLevelFailure: failure.providerLevelFailure } : {}),
                ...(failure.metadata ? { metadata: failure.metadata } : {}),
            };
            trace.push(failed);
            await options.onTrace?.(failed);
        }
    }

    return { ok: false, trace, error: 'no route attempt succeeded' };
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/**
 * Run a single attempt under a timeout / parent-abort race.
 *
 * - No timeout configured: pass the parent signal straight through (or a fresh
 *   never-aborting signal) so the call still observes caller aborts.
 * - Timeout configured: create a child controller; whichever fires first
 *   (the timeout or a parent abort) aborts the child signal and rejects the
 *   race with the corresponding error. The timeout error carries
 *   `name = 'TimeoutError'` and a "timed out after Nms" message so the caller's
 *   error classifier can recognise it.
 */
async function withAttemptTimeout<T>(
    call: (signal: AbortSignal) => Promise<T> | T,
    parentSignal: AbortSignal | undefined,
    timeoutMs: number | undefined,
): Promise<T> {
    if (!timeoutMs || timeoutMs <= 0) {
        return call(parentSignal ?? new AbortController().signal);
    }

    const controller = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let abortFromParent: (() => void) | undefined;

    const abortOrTimeout = new Promise<never>((_, reject) => {
        timeoutId = setTimeout(() => {
            const error = new Error(`Provider attempt timed out after ${timeoutMs}ms`);
            error.name = 'TimeoutError';
            controller.abort(error);
            reject(error);
        }, timeoutMs);
        abortFromParent = () => {
            const error = abortError(parentSignal?.reason);
            controller.abort(error);
            reject(error);
        };
        parentSignal?.addEventListener('abort', abortFromParent, { once: true });
    });

    try {
        return await Promise.race([Promise.resolve(call(controller.signal)), abortOrTimeout]);
    } finally {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
        if (abortFromParent) {
            parentSignal?.removeEventListener('abort', abortFromParent);
        }
    }
}

function normalizeSkip(skip: AttemptSkip | string): AttemptSkip {
    return typeof skip === 'string' ? { reason: skip } : skip;
}

function defaultFailure(error: unknown): AttemptFailure {
    return { message: error instanceof Error ? error.message : String(error) };
}

function abortError(reason: unknown): Error {
    if (reason instanceof Error) {
        return reason;
    }
    return new DOMException('attempt chain aborted', 'AbortError');
}
