// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// Unit tests for the budget/price layer — mirrors the Rust `budget.rs` tests.
// The cross-core parity guard lives in `budget-vectors.test.ts` (shared JSON with
// the Rust `budget_vectors.rs`).

import { describe, expect, it } from "bun:test";
import { Budget, budgetDim, DEFAULT_STEP, MAX_LAMBDA, Tensor, TokenBucket } from "./index";

describe("budget price as a shared weight (not the gas pedal)", () => {
  it("publishes λ as the shared budget-dim weight; a saturated pool sinks its option", () => {
    const t = new Tensor(["global"]);
    {
      const w = t.writer("global")!;
      w.setWeight("global", "", "financial", 1.0);
      w.setValue("global", "", "free_a", "financial", 0.0);
      w.setValue("global", "", "free_b", "financial", 0.0);
      // Each option draws 1 unit on its own pool per resolve.
      w.setValue("global", "", "free_a", budgetDim("acct_a"), 1.0);
      w.setValue("global", "", "free_b", budgetDim("acct_b"), 1.0);
    }
    const budget = new Budget()
      .withPool("acct_a", 100.0)
      .withPool("acct_b", 100.0)
      .mapOption("free_a", "acct_a")
      .mapOption("free_b", "acct_b");

    // Both idle ⇒ both prices 0 ⇒ tie broken by name.
    budget.publishPrices(t.writer("global")!, "global");
    expect(t.resolve({})).toEqual(["free_a", "free_b"]);

    // Saturate pool acct_a via a real tick (usage 600 ≫ cap 100 with α=0.1 ⇒ λ=50).
    budget.tick(new Map([["acct_a", 600.0]]));
    budget.publishPrices(t.writer("global")!, "global");
    // free_a now scores 0 + λ_a·1 = 50; free_b stays 0 ⇒ free_b leads.
    expect(t.resolve({})).toEqual(["free_b", "free_a"]);
  });
});

describe("dual-step price update", () => {
  it("raises price over cap and decays to zero when idle (projected, never negative)", () => {
    const b = new Budget().withStep(0.1).withPool("p", 100.0);
    // Over-consumed: usage 150 > cap 100 ⇒ λ += 0.1·(150-100) = 5.
    b.tick(new Map([["p", 150.0]]));
    expect(b.lambda("p")).toBe(5.0);
    b.tick(new Map([["p", 150.0]]));
    expect(b.lambda("p")).toBe(10.0);
    // Idle window (usage 0 < cap 100): λ ← max(0, 10 + 0.1·(0-100)) = 0.
    b.tick(new Map());
    expect(b.lambda("p")).toBe(0.0);
    // Projection floor: a deeply idle pool never goes negative.
    b.tick(new Map());
    expect(b.lambda("p")).toBe(0.0);
  });

  it("ignores a non-finite measured usage (holds the price)", () => {
    const b = new Budget().withStep(0.1).withPool("p", 10.0);
    b.tick(new Map([["p", 30.0]])); // λ = 0.1·20 = 2
    expect(b.lambda("p")).toBe(2.0);
    b.tick(new Map([["p", Infinity]])); // non-finite → hold
    expect(b.lambda("p")).toBe(2.0);
  });

  it("clamps a runaway λ to the finite ceiling; the saturated option sorts LAST", () => {
    // Without the ceiling, an absurd step on an over-cap pool drives λ to +∞, which
    // resolve SKIPS → the saturated option routes FREE (front). The clamp keeps λ finite
    // (MAX_LAMBDA) and the saturated option sorts LAST. The step is chosen so `next`
    // genuinely OVERFLOWS to +∞ (non-vacuous: dropping the clamp flips the queue).
    const t = new Tensor(["global"]);
    {
      const w = t.writer("global")!;
      w.setWeight("global", "", "financial", 1.0);
      w.setValue("global", "", "saturated", "financial", 0.0);
      w.setValue("global", "", "rival", "financial", 1.0);
      w.setValue("global", "", "saturated", budgetDim("p"), 1.0);
    }
    const b = new Budget().withStep(1e300).withPool("p", 0.0).mapOption("saturated", "p");
    b.tick(new Map([["p", 1e10]])); // next = 1e300·1e10 = 1e310 → OVERFLOWS to +∞ without clamp
    expect(Number.isFinite(b.lambda("p"))).toBe(true);
    expect(b.lambda("p")).toBe(MAX_LAMBDA);
    b.publishPrices(t.writer("global")!, "global");
    expect(t.resolve({})).toEqual(["rival", "saturated"]);
  });
});

describe("saturation simulation — load sheds to the idle pool", () => {
  it("rising hot price sorts its option behind the still-free cold option", () => {
    const t = new Tensor(["global"]);
    {
      const w = t.writer("global")!;
      w.setWeight("global", "", "financial", 1.0);
      w.setValue("global", "", "hot", "financial", 0.0);
      w.setValue("global", "", "cold", "financial", 0.0);
      w.setValue("global", "", "hot", budgetDim("hot_pool"), 1.0);
      w.setValue("global", "", "cold", budgetDim("cold_pool"), 1.0);
    }
    const budget = new Budget()
      .withStep(0.05)
      .withPool("hot_pool", 10.0)
      .withPool("cold_pool", 10.0)
      .mapOption("hot", "hot_pool")
      .mapOption("cold", "cold_pool");

    budget.publishPrices(t.writer("global")!, "global");
    expect(t.resolve({})).toEqual(["cold", "hot"]); // both free, tie by name

    let order: string[] = [];
    for (let win = 0; win < 8; win++) {
      // Fresh per-window usage feeder (the orchestrator's window accumulator, reset
      // per window by live-scoping) so reduceSum sums only this window's draws.
      const feeder = new Tensor(["window", "job"]);
      for (let j = 0; j < 30; j++) {
        budget.writeUsage(feeder.writer("job")!, "job", `j${j}`, "hot", 1.0);
      }
      const measured = budget.aggregateUsage(feeder, "job", "window", "agg");
      expect(measured.get("hot_pool")).toBe(30.0);
      expect(measured.get("cold_pool")).toBe(0.0);
      budget.tick(measured);
      budget.publishPrices(t.writer("global")!, "global");
      order = t.resolve({});
    }
    expect(order).toEqual(["cold", "hot"]);
    expect(budget.lambda("hot_pool")).toBeGreaterThan(0);
    expect(budget.lambda("cold_pool")).toBe(0);
  });
});

describe("usage aggregation (reduceSum feeder)", () => {
  it("sums every draw on a shared pool across instances and options", () => {
    const t = new Tensor(["global", "job"]);
    const budget = new Budget()
      .withPool("acct", 50.0)
      .mapOption("m1", "acct")
      .mapOption("m2", "acct");
    {
      const w = t.writer("job")!;
      budget.writeUsage(w, "job", "j1", "m1", 10.0);
      budget.writeUsage(w, "job", "j2", "m1", 20.0);
      budget.writeUsage(w, "job", "j3", "m2", 5.0);
    }
    const m = budget.aggregateUsage(t, "job", "global", "agg");
    expect(m.get("acct")).toBe(35.0); // 10 + 20 + 5
  });

  it("an unmapped option carries no budget term", () => {
    const budget = new Budget().withPool("p", 10.0);
    const t = new Tensor(["global"]);
    const w = t.writer("global")!;
    expect(budget.writeUsage(w, "global", "", "lonely", 9.0)).toBe(false);
  });

  it("a non-finite usage writes nothing and keeps the option routable", () => {
    // A non-finite usage must NOT land (it would read as a +∞ gate and drop the option).
    // writeUsage returns false and resolve still keeps it; aggregateUsage reports 0.
    const t = new Tensor(["global", "job"]);
    const budget = new Budget().withPool("p", 100.0).mapOption("m", "p");
    {
      const w = t.writer("job")!;
      w.setWeight("job", "", "financial", 1.0);
      w.setValue("job", "j1", "m", "financial", 0.0);
      expect(budget.writeUsage(w, "job", "j1", "m", NaN)).toBe(false);
      expect(budget.writeUsage(w, "job", "j1", "m", Infinity)).toBe(false);
    }
    expect(t.resolve({ job: "j1" })).toEqual(["m"]); // no +∞ gate leaked
    const m = budget.aggregateUsage(t, "job", "global", "agg");
    expect(m.get("p")).toBe(0.0);
  });
});

describe("token-bucket admission gate", () => {
  it("admits a burst, throttles when empty, then refills", () => {
    const tb = new TokenBucket(3.0, 1.0, 0.0);
    expect(tb.admit(0.0)).toBe(true); // 3 → 2
    expect(tb.admit(0.0)).toBe(true); // 2 → 1
    expect(tb.admit(0.0)).toBe(true); // 1 → 0
    expect(tb.admit(0.0)).toBe(false); // empty → denied (the mid-window 429)
    expect(tb.admit(0.5)).toBe(false); // half a token, still < 1
    expect(tb.admit(1.0)).toBe(true); // one token refilled → admitted
    expect(tb.admit(1.0)).toBe(false); // empty again
  });

  it("caps refill at capacity and ignores time going backwards", () => {
    const tb = new TokenBucket(2.0, 5.0, 0.0);
    expect(tb.available(100.0)).toBe(2.0); // 500 would refill but cap is 2
    expect(tb.admit(100.0)).toBe(true); // 2 → 1
    expect(tb.available(50.0)).toBe(1.0); // backwards: no change
  });
});

describe("input validation", () => {
  it("floors an invalid step and clamps an invalid cap", () => {
    const b = new Budget().withStep(0.0).withStep(-1.0); // both rejected
    expect(b.step()).toBe(DEFAULT_STEP);
    const b2 = new Budget().withPool("bad", NaN).withPool("neg", -5.0);
    expect(b2.pool("bad")!.cap).toBe(0.0);
    expect(b2.pool("neg")!.cap).toBe(0.0);
  });
});
