// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Per-router cooldown + dedup + balanced-key-rotation STATE, made injectable.
 *
 * The execution path used to hold these as module globals. They are now a
 * {@link RouterState} object so a host can run an isolated router instance —
 * while `callWithChain` defaults to a single module-level instance, preserving
 * the process-scoped behaviour (cross-call cooldown sharing) and the
 * `clearProviderCooldownForTests` test seam.
 */

const DEDUP_TTL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Mutable router state: provider cooldowns, session-scoped warning dedup, and
 * the balanced multi-key rotation cursor. Construct a fresh one via
 * {@link createRouterState} for an isolated router; omit it to share the module
 * singleton.
 */
export interface RouterState {
    /** providerCooldownKey(provider, model) -> epoch ms the cooldown expires at. */
    readonly cooldown: Map<string, number>;
    /** dedup key -> { count, expiry } for noisy warning suppression. */
    readonly dedup: Map<string, { count: number; expiry: number }>;
    /** provider id -> next rotation cursor for balanced multi-key usage. */
    readonly balancedKeyCursor: Map<string, number>;
}

/** Create a fresh, isolated router state. */
export function createRouterState(): RouterState {
    return {
        cooldown: new Map<string, number>(),
        dedup: new Map<string, { count: number; expiry: number }>(),
        balancedKeyCursor: new Map<string, number>(),
    };
}

/**
 * The default process-scoped state. `callWithChain` and `resolveChain` use this
 * unless a host threads its own {@link RouterState}, so behaviour is identical to
 * the original module-global maps.
 */
export const defaultRouterState: RouterState = createRouterState();

export function providerCooldownKey(providerId: string, model: string): string {
    return `${providerId}|${model}`;
}

/**
 * Session-scoped deduplication for noisy provider warnings (429s, CORS blocks).
 * Tracks {key → count} so the first occurrence logs normally and subsequent
 * identical warnings within the same session are suppressed. Keys expire after
 * DEDUP_TTL_MS to allow fresh warnings after the session cools.
 */
export function shouldSuppressRepeat(state: RouterState, key: string): boolean {
    const now = Date.now();
    const entry = state.dedup.get(key);
    if (!entry || entry.expiry < now) {
        state.dedup.set(key, { count: 1, expiry: now + DEDUP_TTL_MS });
        return false; // first occurrence — do log
    }
    entry.count++;
    return entry.count > 1; // suppress repeats within the TTL window
}

export function activeCooldownSeconds(state: RouterState, providerId: string, model: string): number | undefined {
    const key = providerCooldownKey(providerId, model);
    const until = state.cooldown.get(key);
    if (!until) return undefined;
    const remaining = until - Date.now();
    if (remaining <= 0) {
        state.cooldown.delete(key);
        return undefined;
    }
    return Math.ceil(remaining / 1000);
}

export function setProviderCooldown(state: RouterState, providerId: string, model: string, seconds: number | undefined): void {
    const cooldownSeconds = Math.max(30, Math.min(3600, seconds ?? 300));
    state.cooldown.set(providerCooldownKey(providerId, model), Date.now() + cooldownSeconds * 1000);
}

/** Clear the default state's cooldowns (test seam). */
export function clearProviderCooldownForTests(): void {
    defaultRouterState.cooldown.clear();
}
