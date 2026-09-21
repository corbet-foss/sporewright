// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * trust — the pure cross-device trust / freshness / churn math.
 *
 * This is a PURE, side-effect-free, storage-agnostic module: no SQL, no
 * `Date.now()`, no `Tensor` mutation. It is deliberately NOT a `Tensor` method:
 * the pairwise-increment model is a mutation keyed on a device PAIR, not a level
 * roll-up / reduce, so it does not belong on the Rust-twinned, golden-vector-locked
 * `Tensor` class. It lives here as free functions.
 *
 * The math is relocated VERBATIM from `jobcache/shared/src/fact-tree.ts` (the live
 * pairwise model, jobcache 2026-06). Numbers are byte-identical to that source — the
 * adapter (fact-tree.ts) keeps every DB read/write and delegates the arithmetic here.
 * The DB stays the authority on the persisted value (same contract as `Tensor`: the
 * package exposes math only; the DB versions the write).
 */

// --- Trust constants (relocated verbatim from fact-tree.ts) ---
export const TRUST_AGREE_BASE_GAIN = 0.02;
export const TRUST_DISAGREE_BASE_LOSS = 0.04;
export const TRUST_TIME_HALF_LIFE_HOURS = 48;
/** numberValue(..., 0.5) fallback used when a device has no stored trust_score yet. */
export const TRUST_DEFAULT = 0.5;
/** State-machine thresholds (the SQL CASE in updateDeviceTrust). */
export const TRUST_TRUSTED_AT = 0.8;
export const TRUST_LIMITED_BELOW = 0.3;

// --- Freshness / re-investigation / churn constants (relocated verbatim) ---
export const VERIFY_CONFLICT_MIN_WEIGHT = 0.1;
export const FACT_CHURN_COOLDOWN_THRESHOLD = 6;
export const FACT_CHURN_WINDOW_MS = 86_400_000;
export const FACT_COOLDOWN_MS = 86_400_000;

/**
 * Clamp to [0, 1]. Verbatim from fact-tree.ts:560-563.
 *
 * NOTE: a non-finite input snaps to 0.5 (the trust default), NOT to 0 or 1. This is
 * load-bearing — a "cleanup" to plain `Math.max(0, Math.min(1, x))` would silently
 * change trust behaviour when a bad (e.g. Infinity) value leaks in. Copy as-is.
 */
export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
}

/**
 * Time-decayed evidence weight: closer observations = stronger evidence.
 *
 * Pure form of fact-tree.ts:565-571. The adapter computes the time delta (it owns the
 * `Date.parse` / `Date` handling) and passes `deltaHours` already in hours. A non-finite
 * `deltaHours` (e.g. an unparseable timestamp upstream) returns 0.5, mirroring the
 * source's `if (!Number.isFinite(...)) return 0.5` fallback — keep that fallback in BOTH
 * the adapter's delta calc AND here.
 */
export function timeWeight(deltaHours: number): number {
  if (!Number.isFinite(deltaHours)) return 0.5;
  return Math.pow(2, -Math.abs(deltaHours) / TRUST_TIME_HALF_LIFE_HOURS);
}

export interface PairTrustResult {
  incomingTrust: number;
  currentTrust: number;
}

/**
 * Symmetric pairwise trust increment. Each device's trust moves by the base
 * gain/loss weighted by the OTHER device's current trust and by the freshness weight.
 *
 * Behaviour-identical to fact-tree.ts:2416-2426: `base = agree ? +GAIN : -LOSS`, then
 * `next = clamp01(self + base * other * freshness)` for each side.
 */
export function updatePairTrust(input: {
  incomingTrust: number;
  currentTrust: number;
  agree: boolean;
  freshness: number;
}): PairTrustResult {
  const base = input.agree ? TRUST_AGREE_BASE_GAIN : -TRUST_DISAGREE_BASE_LOSS;
  return {
    incomingTrust: clamp01(input.incomingTrust + base * input.currentTrust * input.freshness),
    currentTrust: clamp01(input.currentTrust + base * input.incomingTrust * input.freshness),
  };
}

export type TrustState =
  | "trusted"
  | "limited"
  | "suspended"
  | "revoked"
  | "probationary"
  | string;

/**
 * Trust state-machine reduction — pure form of the SQL CASE in
 * updateDeviceTrust fact-tree.ts:2446-2451.
 *
 * 'suspended' and 'revoked' are sticky terminal states; otherwise cross the
 * thresholds: >= TRUST_TRUSTED_AT → 'trusted', < TRUST_LIMITED_BELOW → 'limited',
 * else unchanged. (Not yet bound in the relocate — the adapter keeps the SQL CASE
 * authoritative; this is the pure twin for tests / future convergence.)
 */
export function nextTrustState(prev: TrustState, nextScore: number): TrustState {
  if (prev === "suspended" || prev === "revoked") return prev;
  if (nextScore >= TRUST_TRUSTED_AT) return "trusted";
  if (nextScore < TRUST_LIMITED_BELOW) return "limited";
  return prev;
}

/**
 * Freshness-driven re-investigation gate — pure form of shouldQueueFollowUp
 * fact-tree.ts:2320-2329. The adapter supplies the already-resolved churn count, the
 * cooldown deadline as epoch-ms (or null), and `nowMs` (it owns `Date.now()`).
 */
export function shouldReinvestigate(input: {
  freshness: number;
  churnCount: number;
  cooldownUntilMs: number | null;
  nowMs: number;
}): boolean {
  if (input.freshness < VERIFY_CONFLICT_MIN_WEIGHT) return false;
  if (input.churnCount >= FACT_CHURN_COOLDOWN_THRESHOLD) return false;
  return input.cooldownUntilMs == null || input.cooldownUntilMs <= input.nowMs;
}

/**
 * Task priority for a re-investigation — pure form of the literal
 * `Math.max(1, Math.round(1000 * freshness))` at fact-tree.ts:2376.
 */
export function reinvestigationPriority(freshness: number): number {
  return Math.max(1, Math.round(1_000 * freshness));
}

/**
 * Provenance of a churn timestamp output. `'carried'` means the value is the prior
 * stored timestamp unchanged — the adapter should re-emit its stored value VERBATIM
 * (the live model carried these via `dateString(...)`, preserving the exact stored
 * string, not a re-canonicalized one). `'fresh'` means the value is newly computed from
 * the observation time and the adapter serializes `valueMs`. `'observed'` means the value
 * is the observation time itself (used as the window fallback when there is no prior
 * window-start). This discriminant exists purely so the adapter reproduces byte-identical
 * bound parameters; the arithmetic decision (which source each output comes from) is made
 * here in the pure function.
 */
export interface ChurnTimestamp {
  /** Numeric epoch-ms value of this timestamp (null only for an absent cooldown). */
  valueMs: number | null;
  /** Where the value came from — drives byte-identical serialization in the adapter. */
  source: "carried" | "fresh" | "observed";
}

export interface ChurnState {
  count: number;
  /** Epoch-ms of the window start (always present). */
  windowStartedAtMs: number;
  /** Epoch-ms of the cooldown deadline, or null when none is set. */
  cooldownUntilMs: number | null;
  /** Provenance of windowStartedAtMs — `'carried'` | `'fresh'` | `'observed'`. */
  windowStartedAt: ChurnTimestamp;
  /** Provenance of cooldownUntilMs — `'carried'` (incl. null carry) | `'fresh'`. */
  cooldownUntil: ChurnTimestamp;
}

/**
 * Churn-window roll/reset/cooldown — pure form of nextChurn fact-tree.ts:2294-2318.
 *
 * The adapter owns all timestamp parsing and serialization (Date ↔ epoch-ms). It passes
 * the previous state as epoch-ms (or null if the field has no prior state) and the new
 * observation time as epoch-ms; this returns the next state in epoch-ms PLUS a provenance
 * discriminant per timestamp so the adapter can serialize byte-identically to the live
 * model (carried values re-emit the stored string verbatim; fresh values serialize the ms;
 * the window fallback uses the observation time).
 *
 * - `same` (the new observation matches the stored value): hold the existing window;
 *   carry prior count / window-start / cooldown unchanged (count defaults to 0, the
 *   window-start falls back to the observation time, cooldown to null when there is no
 *   prior state — matching `dateString(...) ?? observedAt.toISOString()` / `?? null`).
 * - changed: if the prior window is still active (within FACT_CHURN_WINDOW_MS), increment
 *   the count and keep the window start (re-emitted fresh, as the live model did with
 *   `windowStart.toISOString()`); otherwise start a fresh window of count 1. When the count
 *   reaches FACT_CHURN_COOLDOWN_THRESHOLD, set a cooldown FACT_COOLDOWN_MS out (fresh);
 *   otherwise carry the prior cooldown forward verbatim.
 */
export function nextChurn(input: {
  prev: {
    count: number;
    /**
     * Parsed epoch-ms of the stored window-start (null if the stored value is absent OR
     * unparseable). Drives the active-window arithmetic — mirrors the live model's
     * `dateValue(...)` in the changed branch.
     */
    windowStartedAtMs: number | null;
    /**
     * Whether a stored window-start string/Date is PRESENT (truthy), regardless of
     * parseability. Drives verbatim carry in the `same` branch — mirrors the live model's
     * `dateString(...)` (a non-empty unparseable string is carried verbatim, not parsed).
     */
    hasStoredWindow: boolean;
    cooldownUntilMs: number | null;
  } | null;
  same: boolean;
  observedAtMs: number;
}): ChurnState {
  const { prev, same, observedAtMs } = input;

  if (same) {
    // windowStartedAt: a present stored window-start is carried verbatim (even if
    // unparseable), else the observation time — matching `dateString(...) ?? observedAt`.
    const window: ChurnTimestamp = prev?.hasStoredWindow
      ? { valueMs: prev.windowStartedAtMs, source: "carried" }
      : { valueMs: observedAtMs, source: "observed" };
    return {
      count: prev?.count ?? 0,
      windowStartedAtMs: window.valueMs ?? observedAtMs,
      cooldownUntilMs: prev?.cooldownUntilMs ?? null,
      windowStartedAt: window,
      cooldownUntil: { valueMs: prev?.cooldownUntilMs ?? null, source: "carried" },
    };
  }

  const windowStartMs = prev?.windowStartedAtMs ?? null;
  const activeWindow = windowStartMs != null && observedAtMs - windowStartMs <= FACT_CHURN_WINDOW_MS;
  const count = activeWindow ? (prev?.count ?? 0) + 1 : 1;
  // The live model re-canonicalizes the active window start (`windowStart.toISOString()`)
  // and uses observedAt for a fresh window, so both are 'fresh' (serialize the ms).
  const startedAtMs = activeWindow ? windowStartMs! : observedAtMs;
  const cooldownMs = count >= FACT_CHURN_COOLDOWN_THRESHOLD ? observedAtMs + FACT_COOLDOWN_MS : prev?.cooldownUntilMs ?? null;
  return {
    count,
    windowStartedAtMs: startedAtMs,
    cooldownUntilMs: cooldownMs,
    windowStartedAt: { valueMs: startedAtMs, source: "fresh" },
    cooldownUntil:
      count >= FACT_CHURN_COOLDOWN_THRESHOLD
        ? { valueMs: cooldownMs, source: "fresh" }
        : { valueMs: cooldownMs, source: "carried" },
  };
}
