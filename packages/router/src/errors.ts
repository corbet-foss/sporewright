// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Public execution types + error classes for the LLM fallback engine.
 *
 * All generic to the Vercel AI SDK — no host/persistence vocabulary. The
 * host-coupled telemetry concerns (a gateway sink, workspace/job ids) are NOT
 * here: the engine emits an {@link import('./attempt-chain').AttemptTrace} for
 * every attempt and the host wires whatever observer it likes via `onTrace`.
 */

import type { RouterState } from './cooldown';

/** A resolved routing slot: a provider id, an API key, and a model. */
export interface ProviderSlot {
    providerId: string;
    apiKey: string;
    model: string;
}

/** A per-provider failure record for transparent error reporting. */
export interface ProviderFailure {
    providerId: string;
    model: string;
    error: string;
    statusCode?: number;
}

/** Token usage for one successful call. */
export interface UsageInfo {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
}

export type ChainAttemptStatus = 'selected' | 'rejected' | 'failed';

/** One classified attempt in the chain (the host-facing view of an AttemptTrace). */
export interface ChainAttempt {
    attemptNo: number;
    providerId: string;
    model: string;
    status: ChainAttemptStatus;
    latencyMs: number;
    wasFallback: boolean;
    usage?: UsageInfo;
    error?: string;
    statusCode?: number;
    retryAfterSeconds?: number;
    criticRejected?: boolean;
}

/** The result of a successful chain call. */
export interface CallResult<T> {
    result: T;
    providerId: string;
    model: string;
    wasFallback: boolean;
    usage?: UsageInfo;
    attempts: ChainAttempt[];
}

export type ChainAttemptObserver = (attempt: ChainAttempt) => void | Promise<void>;

/** Minimal injectable logger. Default is a no-op (router carries no build-tool assumption). */
export interface RouterLogger {
    log: (msg: string) => void;
    warn: (msg: string) => void;
}

export interface CallWithChainOptions<T = unknown> {
    critic?: (result: T) => boolean;
    label?: string;
    /**
     * Maximum wall-clock time for one provider/model attempt. A timeout is
     * recorded as a failed attempt and the chain continues to the next slot.
     */
    attemptTimeoutMs?: number;
    /**
     * Called after each provider attempt is classified. This is intentionally
     * best-effort so telemetry persistence cannot break the model fallback
     * path that produced the attempt.
     */
    onAttempt?: ChainAttemptObserver;
    /**
     * Best-effort observer fired for every emitted {@link import('./attempt-chain').AttemptTrace}.
     * The host wires telemetry here (e.g. a gateway sink). Must never throw in a
     * way that breaks routing; the engine does not await this beyond the trace
     * emission contract.
     */
    onTrace?: (trace: import('./attempt-chain').AttemptTrace<ProviderSlot>) => void;
    /**
     * Injected logger. Defaults to a no-op so the router has no Vite/build-tool
     * assumption (no `import.meta.env`). The host supplies a dev/tier-aware logger.
     */
    logger?: RouterLogger;
    /**
     * Per-router cooldown / dedup / balanced-key-rotation state. Defaults to a
     * module singleton so process-scoped behaviour (and the test-clear seam) is
     * preserved; pass a fresh state for an isolated router instance.
     */
    state?: RouterState;
    /**
     * AbortSignal forwarded into each provider attempt. When the signal fires
     * mid-chain, the current `callFn` call receives the same signal (if the
     * underlying `generateText` honours it) and any pending retries are skipped.
     */
    signal?: AbortSignal;
}

/**
 * Thrown when all providers in a chain are rate-limited.
 * Carries retry timing so callers can return proper 429 responses.
 */
export class RateLimitError extends Error {
    readonly retryAfterSeconds: number | undefined;
    readonly providers: string[];
    readonly attempts: ChainAttempt[];

    constructor(providers: string[], retryAfterSeconds?: number, attempts: ChainAttempt[] = []) {
        const msg = `All providers rate-limited (${providers.join(', ')}). ${retryAfterSeconds ? `Retry after ${retryAfterSeconds}s.` : 'Try again later.'}`;
        super(msg);
        this.name = 'RateLimitError';
        this.providers = providers;
        this.retryAfterSeconds = retryAfterSeconds;
        this.attempts = attempts;
    }
}

/**
 * Thrown when all providers in a chain fail (not all necessarily rate-limited).
 * Carries per-provider failure details for transparent error reporting.
 */
export class ChainExhaustionError extends Error {
    readonly failures: ProviderFailure[];
    readonly attempts: ChainAttempt[];

    constructor(failures: ProviderFailure[], attempts: ChainAttempt[] = []) {
        const summary = failures
            .map(f => `${f.providerId}${f.statusCode ? ` (${f.statusCode})` : ''}`)
            .join(', ');
        super(`All ${failures.length} providers failed: ${summary}`);
        this.name = 'ChainExhaustionError';
        this.failures = failures;
        this.attempts = attempts;
    }
}
