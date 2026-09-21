// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { expect, test } from "bun:test";
import {
  FACT_CHURN_COOLDOWN_THRESHOLD,
  FACT_CHURN_WINDOW_MS,
  FACT_COOLDOWN_MS,
  TRUST_AGREE_BASE_GAIN,
  TRUST_DEFAULT,
  TRUST_DISAGREE_BASE_LOSS,
  TRUST_LIMITED_BELOW,
  TRUST_TIME_HALF_LIFE_HOURS,
  TRUST_TRUSTED_AT,
  VERIFY_CONFLICT_MIN_WEIGHT,
  clamp01,
  nextChurn,
  nextTrustState,
  reinvestigationPriority,
  shouldReinvestigate,
  timeWeight,
  updatePairTrust,
} from "./index";

// These golden values are the EXACT numbers the live jobcache pairwise model produced
// (fact-tree.ts, pre-relocate). They are the behaviour-preservation contract for the
// relocate: any drift here means the relocated math diverged from the source.

test("constants are the verbatim live values", () => {
  expect(TRUST_AGREE_BASE_GAIN).toBe(0.02);
  expect(TRUST_DISAGREE_BASE_LOSS).toBe(0.04);
  expect(TRUST_TIME_HALF_LIFE_HOURS).toBe(48);
  expect(TRUST_DEFAULT).toBe(0.5);
  expect(TRUST_TRUSTED_AT).toBe(0.8);
  expect(TRUST_LIMITED_BELOW).toBe(0.3);
  expect(VERIFY_CONFLICT_MIN_WEIGHT).toBe(0.1);
  expect(FACT_CHURN_COOLDOWN_THRESHOLD).toBe(6);
  expect(FACT_CHURN_WINDOW_MS).toBe(86_400_000);
  expect(FACT_COOLDOWN_MS).toBe(86_400_000);
});

test("clamp01 — the non-finite→0.5 quirk is load-bearing", () => {
  expect(clamp01(0.5)).toBe(0.5);
  expect(clamp01(2)).toBe(1);
  expect(clamp01(-1)).toBe(0);
  expect(clamp01(0)).toBe(0);
  expect(clamp01(1)).toBe(1);
  // non-finite snaps to 0.5, NOT 0 or 1
  expect(clamp01(Infinity)).toBe(0.5);
  expect(clamp01(-Infinity)).toBe(0.5);
  expect(clamp01(NaN)).toBe(0.5);
  // idempotent (the adapter double-clamps; this must be a no-op)
  expect(clamp01(clamp01(2))).toBe(1);
  expect(clamp01(clamp01(Infinity))).toBe(0.5);
});

test("timeWeight — half-life decay, NaN→0.5", () => {
  expect(timeWeight(0)).toBe(1); // 2^0
  expect(timeWeight(48)).toBe(0.5); // 2^-1 at one half-life
  expect(timeWeight(96)).toBe(0.25); // 2^-2 at two half-lives
  expect(timeWeight(24)).toBeCloseTo(Math.SQRT1_2, 12); // 2^-0.5
  expect(timeWeight(-48)).toBe(0.5); // abs() — sign-symmetric
  expect(timeWeight(NaN)).toBe(0.5);
  expect(timeWeight(Infinity)).toBe(0.5);
});

test("updatePairTrust — symmetric increment at default 0.5/0.5", () => {
  // agree at 0.5/0.5 freshness 1.0 → 0.51/0.51
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: true, freshness: 1.0 })).toEqual({
    incomingTrust: 0.51,
    currentTrust: 0.51,
  });
  // disagree → 0.48/0.48
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: false, freshness: 1.0 })).toEqual({
    incomingTrust: 0.48,
    currentTrust: 0.48,
  });
  // freshness scales the move: agree at freshness 0.5 → +0.005 each
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: true, freshness: 0.5 })).toEqual({
    incomingTrust: 0.505,
    currentTrust: 0.505,
  });
  // freshness 0 → no movement
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: true, freshness: 0 })).toEqual({
    incomingTrust: 0.5,
    currentTrust: 0.5,
  });
});

test("updatePairTrust — asymmetric trust weights each side by the OTHER's trust", () => {
  // incoming=0.5, current=0.8, agree, freshness=1: base=+0.02
  //   incoming' = clamp01(0.5 + 0.02*0.8*1) = 0.516
  //   current'  = clamp01(0.8 + 0.02*0.5*1) = 0.81
  const r = updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.8, agree: true, freshness: 1.0 });
  expect(r.incomingTrust).toBeCloseTo(0.516, 12);
  expect(r.currentTrust).toBeCloseTo(0.81, 12);
});

test("updatePairTrust — clamps at the rails", () => {
  // huge freshness on disagree drives below 0 → clamped to 0
  expect(updatePairTrust({ incomingTrust: 0.01, currentTrust: 1, agree: false, freshness: 100 }).incomingTrust).toBe(0);
  // agree near 1 with big push → clamped to 1
  expect(updatePairTrust({ incomingTrust: 0.99, currentTrust: 1, agree: true, freshness: 100 }).incomingTrust).toBe(1);
});

test("updatePairTrust — source volatility discounts disagreement only", () => {
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: false, freshness: 1, sourceVolatility: 0.75 }))
    .toEqual({ incomingTrust: 0.495, currentTrust: 0.495 });
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: false, freshness: 1, sourceVolatility: 1 }))
    .toEqual({ incomingTrust: 0.5, currentTrust: 0.5 });
  expect(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: true, freshness: 1, sourceVolatility: 1 }))
    .toEqual(updatePairTrust({ incomingTrust: 0.5, currentTrust: 0.5, agree: true, freshness: 1 }));
});

test("nextTrustState — sticky terminals and threshold crossings", () => {
  expect(nextTrustState("suspended", 0.95)).toBe("suspended"); // sticky
  expect(nextTrustState("revoked", 0.95)).toBe("revoked"); // sticky
  expect(nextTrustState("probationary", 0.8)).toBe("trusted"); // >= 0.8
  expect(nextTrustState("probationary", 0.85)).toBe("trusted");
  expect(nextTrustState("trusted", 0.2)).toBe("limited"); // < 0.3
  expect(nextTrustState("limited", 0.5)).toBe("limited"); // mid-band → unchanged
  expect(nextTrustState("trusted", 0.5)).toBe("trusted"); // mid-band → unchanged
  // exact boundaries: 0.8 is trusted (>=), 0.3 is NOT limited (< only)
  expect(nextTrustState("probationary", 0.8)).toBe("trusted");
  expect(nextTrustState("probationary", 0.3)).toBe("probationary");
  expect(nextTrustState("probationary", 0.2999999)).toBe("limited");
});

test("shouldReinvestigate — the three gates", () => {
  const base = { freshness: 0.5, churnCount: 0, cooldownUntilMs: null, nowMs: 1_000 };
  expect(shouldReinvestigate(base)).toBe(true);
  // gate 1: below min weight
  expect(shouldReinvestigate({ ...base, freshness: 0.09 })).toBe(false);
  expect(shouldReinvestigate({ ...base, freshness: 0.1 })).toBe(true); // boundary: >= min passes
  // gate 2: churn threshold reached
  expect(shouldReinvestigate({ ...base, churnCount: 6 })).toBe(false);
  expect(shouldReinvestigate({ ...base, churnCount: 5 })).toBe(true);
  // gate 3: cooldown not yet passed
  expect(shouldReinvestigate({ ...base, cooldownUntilMs: 2_000, nowMs: 1_000 })).toBe(false);
  expect(shouldReinvestigate({ ...base, cooldownUntilMs: 1_000, nowMs: 1_000 })).toBe(true); // <= now passes
  expect(shouldReinvestigate({ ...base, cooldownUntilMs: 999, nowMs: 1_000 })).toBe(true);
});

test("reinvestigationPriority — Math.max(1, round(1000*freshness))", () => {
  expect(reinvestigationPriority(0.0005)).toBe(1); // round(0.5)=1 (and floor at 1)
  expect(reinvestigationPriority(0.0004)).toBe(1); // round(0.4)=0 → floored to 1
  expect(reinvestigationPriority(0)).toBe(1); // floored to 1
  expect(reinvestigationPriority(1.0)).toBe(1000);
  expect(reinvestigationPriority(0.5)).toBe(500);
  expect(reinvestigationPriority(0.1)).toBe(100);
});

// --- nextChurn ---
const T0 = Date.parse("2026-06-08T00:00:00.000Z");

test("nextChurn — same with no prior state holds a fresh window of count 0", () => {
  const r = nextChurn({ prev: null, same: true, observedAtMs: T0 });
  expect(r.count).toBe(0);
  expect(r.windowStartedAtMs).toBe(T0);
  expect(r.windowStartedAt.source).toBe("observed");
  expect(r.cooldownUntilMs).toBe(null);
  expect(r.cooldownUntil.source).toBe("carried");
});

test("nextChurn — same with prior state carries everything verbatim", () => {
  const prev = { count: 3, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: true, observedAtMs: T0 + 5_000 });
  expect(r.count).toBe(3); // unchanged
  expect(r.windowStartedAtMs).toBe(T0);
  expect(r.windowStartedAt.source).toBe("carried"); // re-emit stored verbatim
  expect(r.cooldownUntilMs).toBe(null);
  expect(r.cooldownUntil.source).toBe("carried");
});

test("nextChurn — changed inside the active window increments and keeps the start", () => {
  const prev = { count: 2, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 + 1_000 });
  expect(r.count).toBe(3);
  expect(r.windowStartedAtMs).toBe(T0); // window held
  expect(r.windowStartedAt.source).toBe("fresh"); // live model re-canonicalized this
  expect(r.cooldownUntilMs).toBe(null);
  expect(r.cooldownUntil.source).toBe("carried");
});

test("nextChurn — changed past the window resets to a fresh window of count 1", () => {
  const prev = { count: 5, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 + FACT_CHURN_WINDOW_MS + 1 });
  expect(r.count).toBe(1);
  expect(r.windowStartedAtMs).toBe(T0 + FACT_CHURN_WINDOW_MS + 1);
  expect(r.windowStartedAt.source).toBe("fresh");
});

test("nextChurn — exactly at the window edge is still active (<=)", () => {
  const prev = { count: 1, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 + FACT_CHURN_WINDOW_MS });
  expect(r.count).toBe(2); // <= window → active
});

test("nextChurn — reaching the threshold sets a fresh cooldown", () => {
  const prev = { count: 5, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 + 1_000 });
  expect(r.count).toBe(6); // reaches FACT_CHURN_COOLDOWN_THRESHOLD
  expect(r.cooldownUntilMs).toBe(T0 + 1_000 + FACT_COOLDOWN_MS);
  expect(r.cooldownUntil.source).toBe("fresh");
});

test("nextChurn — below threshold carries the prior cooldown forward", () => {
  const cooldown = T0 + 999;
  const prev = { count: 1, windowStartedAtMs: T0, hasStoredWindow: true, cooldownUntilMs: cooldown };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 + 1_000 });
  expect(r.count).toBe(2);
  expect(r.cooldownUntilMs).toBe(cooldown);
  expect(r.cooldownUntil.source).toBe("carried");
});

test("nextChurn — changed with no parsed window starts a fresh window", () => {
  // hasStoredWindow true but windowStartedAtMs null (stored-but-unparseable): the changed
  // branch keys on the PARSED ms (dateValue), so a null parse → inactive → count 1.
  const prev = { count: 9, windowStartedAtMs: null, hasStoredWindow: true, cooldownUntilMs: null };
  const r = nextChurn({ prev, same: false, observedAtMs: T0 });
  expect(r.count).toBe(1);
  expect(r.windowStartedAtMs).toBe(T0);
  expect(r.windowStartedAt.source).toBe("fresh");
});
