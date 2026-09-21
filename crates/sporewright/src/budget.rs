// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! The **budget / price** layer — coupling constraints, priced as shared weights.
//!
//! Some quantities are not judgements about one task; they are **shared, finite
//! pools that many tasks draw down together** (a provider account's daily quota, a
//! target's per-IP rate). The decision for task A depends on what B, C, D already
//! consumed this window — a *coupling* constraint that a per-task greedy choice
//! cannot price in isolation (`ROUTING-MODEL.md` §6).
//!
//! The standard decoupling is **Lagrangian dual decomposition**: relax each budget
//! constraint into the objective with a **shadow price** `λ_pool ≥ 0`, the marginal
//! value of one more unit of the scarce pool. The elegant tensor form **needs no
//! engine change**:
//!
//! * a pool's usage rides as the **value** on a per-task `budget:<pool>` dim
//!   (`usage_o`, the draw this option puts on the pool);
//! * the pool's price `λ_pool` rides as the **shared weight** on that same dim
//!   ([`Writer::set_weight`] at `option=""`, inherited by every contender).
//!
//! So `resolve`'s `Σ value·weight` *already* contributes the term `λ_pool · usage_o`
//! to every option drawing on the pool — exactly the relaxed objective. As a pool
//! saturates its `λ_pool` rises, its contribution to every drawing option's score
//! rises, and load **self-rebalances** away from the scarce pool. No rebalancing
//! rule is written; the shadow price does the job (`ROUTING-MODEL.md` Story 1).
//!
//! **This is the additive price term, not the gas pedal.** The gas pedal
//! ([`Writer::set_option_weight`]) is a *per-option, multiplicative* override that
//! scales one option's existing judgement score — a different mechanism. The price
//! is an *additive* `λ·usage` term shared across every option in the pool. Using the
//! gas pedal here would be wrong: it would multiply a judgement, not add a coupling
//! cost.
//!
//! Three pieces live here:
//! 1. [`Budget`] — the pools `{cap, lambda}` and the option→pool mapping, plus the
//!    `budget:<pool>` dim naming and the helpers to write usage / publish prices.
//! 2. [`Budget::tick`] — the **single-writer** slow-tick price update:
//!    `λ ← max(0, λ + α·(usage − cap))`, an **online projected subgradient** with a
//!    small *constant/floored* step (`[·]₊` enforces `λ ≥ 0`). Quotas refill, so
//!    there is no static optimum — we target **dynamic regret** with a constant
//!    step, never the textbook diminishing one. The accumulator is **non-idempotent**
//!    (`λ` carries window to window) and the substrate is last-write-wins, so this
//!    step is a centralised hub, never distributed (devices only *read* prices).
//! 3. [`TokenBucket`] — a **separate, edge-local token-bucket admission gate**: the
//!    daily-quota price models a *cumulative* cap and stays slack for a purely
//!    *rate*-limited tier, so it cannot stop a mid-window `429`. The bucket is the
//!    sub-second within-window rate limiter that does (`ROUTING-MODEL.md` §6 "trap").
//!
//! Usage is aggregated with [`Tensor::reduce_sum`] (the additive feeder) — **not**
//! `reduce_median`: total draw on a pool is a sum. Decision-equivalent with the TS
//! `budget.ts`.

use crate::tensor::{Scope, Tensor, WriteError, Writer};
use std::collections::BTreeMap;

/// The budget-dim prefix: a pool's per-task usage and shared price live on
/// `budget:<pool>`. The **value** is `usage_o` (this option's draw); the **shared
/// weight** is `λ_pool` (the price). `resolve` then adds `λ_pool · usage_o` to every
/// drawing option's score — the relaxed Lagrangian term, for free.
pub const BUDGET_PREFIX: &str = "budget:";

/// Compose the budget dimension `budget:<pool>` for a resource pool.
pub fn budget_dim(pool: &str) -> String {
    format!("{BUDGET_PREFIX}{pool}")
}

/// One resource pool: a shared finite cap and its current shadow price.
#[derive(Clone, Debug, PartialEq)]
pub struct Pool {
    /// The window cap — the ceiling on summed usage before the price starts rising.
    pub cap: f64,
    /// The current shadow price `λ ≥ 0` (the budget dim's shared weight). Starts at
    /// `0` (an idle pool is free) and rises as usage exceeds the cap. Read-only from
    /// outside the module: external reads go through [`Budget::lambda`], and only
    /// [`Budget::tick`] (in-module) mutates it — so "price changes only via tick" is a
    /// structural guarantee, not a convention.
    pub(crate) lambda: f64,
}

/// The budget book: the pools `{cap, lambda}` keyed by pool id, and the
/// option→pool mapping. A single hub owns this and advances the prices on the slow
/// tick; devices read the published prices off the tensor and never mutate them.
///
/// The explicit [`Default`] implementation delegates to [`Budget::new`] so the
/// default step stays decision-equivalent with the TypeScript core.
#[derive(Clone, Debug, PartialEq)]
pub struct Budget {
    pools: BTreeMap<String, Pool>,
    /// option → the pool it draws on. An option absent from the map draws on no
    /// pool: it carries no budget term and is priced on judgement alone.
    option_pool: BTreeMap<String, String>,
    /// The subgradient step `α > 0` — a small **constant** (dynamic-regret target),
    /// floored away from zero so the price always reacts. Never diminishing.
    step: f64,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new()
    }
}

/// The default constant step: small enough not to overshoot, floored so the price
/// keeps reacting to a refilling quota. A caller may override via [`Budget::with_step`].
pub const DEFAULT_STEP: f64 = 0.1;

/// The smallest step we permit — a floor that guarantees the price never freezes
/// (the dynamic-regret guarantee: a constant, never-diminishing reaction).
pub const MIN_STEP: f64 = 1e-6;

/// The finite ceiling on the shadow price `λ`. A runaway/corrupt step (a huge
/// `with_step` on a pool far over cap, or a poisoned usage feeder) could otherwise drive
/// `λ` to `+∞`; since `resolve` SKIPS a non-finite weight (`if w.is_finite()`), a
/// `+∞`-priced — maximally-saturated — option would score as FREE and sort to the FRONT,
/// the exact INVERSION of the intended shed. Clamping to a large but finite ceiling keeps
/// an over-saturated option sorting LAST. `1e300` sits comfortably above every realistic
/// price (and above the `~1.1e16` the feeder golden pins) yet stays finite, so it sinks
/// any option without ever becoming `+∞`.
pub const MAX_LAMBDA: f64 = 1e300;

impl Budget {
    /// An empty budget book with the [`DEFAULT_STEP`].
    pub fn new() -> Self {
        Budget {
            pools: BTreeMap::new(),
            option_pool: BTreeMap::new(),
            step: DEFAULT_STEP,
        }
    }

    /// Override the constant subgradient step `α`. A non-finite or non-positive step
    /// is rejected (the step stays at its prior value, floored at [`MIN_STEP`]) — a
    /// `0` or negative step would freeze or invert the price, the exact bug the
    /// dynamic-regret constant-step rule exists to prevent.
    pub fn with_step(mut self, step: f64) -> Self {
        if step.is_finite() && step >= MIN_STEP {
            self.step = step;
        }
        self
    }

    /// The effective step `α` (always finite and `≥ MIN_STEP`).
    pub fn step(&self) -> f64 {
        if self.step.is_finite() && self.step >= MIN_STEP {
            self.step
        } else {
            MIN_STEP
        }
    }

    /// Declare a pool with its window `cap`, price starting at `0` (idle ⇒ free). A
    /// non-finite or negative cap is clamped to `0` (a degenerate cap of `0` simply
    /// prices any positive usage immediately, never a non-finite leak).
    pub fn with_pool(mut self, pool: &str, cap: f64) -> Self {
        let cap = if cap.is_finite() && cap >= 0.0 {
            cap
        } else {
            0.0
        };
        self.pools
            .insert(pool.to_string(), Pool { cap, lambda: 0.0 });
        self
    }

    /// Map an `option` onto the `pool` it draws from. Drawing on a pool means the
    /// option carries a `budget:<pool>` usage value and inherits that pool's price.
    pub fn map_option(mut self, option: &str, pool: &str) -> Self {
        self.option_pool
            .insert(option.to_string(), pool.to_string());
        self
    }

    /// The pool an `option` draws on, if any.
    pub fn pool_of(&self, option: &str) -> Option<&str> {
        self.option_pool.get(option).map(String::as_str)
    }

    /// A pool's current `{cap, lambda}`, if declared.
    pub fn pool(&self, pool: &str) -> Option<&Pool> {
        self.pools.get(pool)
    }

    /// The current shadow price `λ` for a pool (`0.0` if undeclared — an unknown
    /// pool is free, never a panic).
    pub fn lambda(&self, pool: &str) -> f64 {
        self.pools.get(pool).map(|p| p.lambda).unwrap_or(0.0)
    }

    /// The pool ids, in canonical order.
    pub fn pools(&self) -> impl Iterator<Item = &str> {
        self.pools.keys().map(String::as_str)
    }

    /// **Write a task's usage** for an option onto its pool's `budget:<pool>` dim, as
    /// a per-task **value** at `(level, inst, option)`. The summed draw on the pool is
    /// later rolled up with [`Tensor::reduce_sum`]. An option that maps to no pool
    /// writes nothing (it carries no budget term).
    ///
    /// `writer` must be able to write at `level` (write-down). Returns `Ok(false)`
    /// when the option maps to no pool (nothing written), `Ok(true)` when the usage
    /// landed, or the first [`WriteError`].
    pub fn write_usage(
        &self,
        writer: &mut Writer<'_>,
        level: &str,
        scope: &Scope,
        option: &str,
        usage: f64,
    ) -> Result<bool, WriteError> {
        let Some(pool) = self.pool_of(option) else {
            return Ok(false);
        };
        // Guard a non-finite usage: a `budget:<pool>` cell is a VALUE cell, and resolve
        // DROPS any option carrying a non-finite value on any dim (it reads as a +∞
        // gate). Writing a NaN/Infinity usage would silently gate the option out of
        // routing instead of pricing its draw — a budget value leaking into the gate
        // role. Write NOTHING (absence over a fabricated 0): the option stays routable,
        // priced on judgement alone, and the pool sees this draw as absent.
        if !usage.is_finite() {
            return Ok(false);
        }
        let dim = budget_dim(pool);
        writer.set_value(level, scope, option, &dim, usage)?;
        Ok(true)
    }

    /// **Publish every pool's price** onto the tensor as the **shared weight** on its
    /// `budget:<pool>` dim (`option=""`), so `resolve` adds `λ_pool · usage_o` to
    /// every option drawing on the pool. This is the additive coupling term — written
    /// via [`Writer::set_weight`] (shared), **never** [`Writer::set_option_weight`]
    /// (the per-option multiplicative gas pedal).
    ///
    /// Single-writer: only the hub calls this, on the slow tick, after [`Budget::tick`].
    pub fn publish_prices(
        &self,
        writer: &mut Writer<'_>,
        level: &str,
        scope: &Scope,
    ) -> Result<(), WriteError> {
        for (pool, p) in &self.pools {
            let dim = budget_dim(pool);
            // Defence-in-depth: never publish a non-finite price (a `+∞` weight is
            // SKIPPED by resolve and would route a saturated option FREE). `tick`
            // already clamps to MAX_LAMBDA, but a directly-set λ must be coerced too.
            let lambda = if p.lambda.is_finite() {
                p.lambda
            } else {
                MAX_LAMBDA
            };
            writer.set_weight(level, scope, &dim, lambda)?;
        }
        Ok(())
    }

    /// **The slow-tick dual step (single-writer).** For each pool, read its summed
    /// usage this window from `measured` and nudge the price toward clearing the cap:
    ///
    /// ```text
    ///   λ ← max(0, λ + α·(usage − cap))
    /// ```
    ///
    /// an **online projected subgradient** with the constant step `α` ([`Budget::step`])
    /// and the `[·]₊` projection (`λ ≥ 0`). Over-consumed (`usage > cap`) ⇒ the price
    /// rises; idle (`usage < cap`) ⇒ it falls back toward `0`. Because quotas refill
    /// there is no static optimum — the constant step targets **dynamic regret**, never
    /// a diminishing schedule. A pool absent from `measured` is treated as `0` usage
    /// (idle ⇒ price decays). A non-finite measured usage is ignored (the pool's price
    /// holds) — a gate leak must never poison the accumulator.
    pub fn tick(&mut self, measured: &BTreeMap<String, f64>) {
        let alpha = self.step();
        for (pool, p) in self.pools.iter_mut() {
            let usage = match measured.get(pool) {
                Some(u) if u.is_finite() => *u,
                Some(_) => continue, // non-finite measured usage → hold the price.
                None => 0.0,         // unmeasured ⇒ idle this window.
            };
            let next = p.lambda + alpha * (usage - p.cap);
            // Clamp the new price to a FINITE ceiling: `[·]₊` keeps λ ≥ 0, MAX_LAMBDA
            // keeps it from ever reaching `+∞` (a `+∞` weight is skipped by resolve and
            // would route a saturated option FREE — the opposite of the intended shed).
            p.lambda = if next.is_finite() && next > 0.0 {
                next.min(MAX_LAMBDA)
            } else if next > 0.0 {
                MAX_LAMBDA // a non-finite-but-positive overshoot pins to the ceiling.
            } else {
                0.0
            };
        }
    }

    /// **Aggregate this window's usage** from the tensor: for each pool, sum the
    /// per-task `budget:<pool>` usage values across every instance at `from_level`
    /// via [`Tensor::reduce_sum`] (which returns each option's summed draw, also
    /// writing it at `to_level`/`to_inst`), accumulated into the `measured` map
    /// [`Budget::tick`] consumes.
    ///
    /// `reduce_sum` is the **additive** feeder (`MODEL.md` §6 / `ROUTING-MODEL.md`
    /// §6): a budget price needs the *sum* of draws, not the median. A pool with no
    /// usage this window reports `0` (idle), so its price decays on the next tick.
    pub fn aggregate_usage(
        &self,
        t: &mut Tensor,
        from_level: &str,
        to_level: &str,
        target_scope: &Scope,
    ) -> BTreeMap<String, f64> {
        let mut measured = BTreeMap::new();
        for pool in self.pools.keys() {
            let dim = budget_dim(pool);
            // Sum every option that draws on this pool. Each drawing option carries
            // its own `budget:<pool>` usage cell; reduce_sum adds them per option, so
            // the pool total is the sum of those per-option sums.
            //
            // **The per-option sum order is load-bearing for cross-core equivalence.**
            // `option_pool` is a `BTreeMap`, so this iterates options SORTED by name —
            // the TS core sorts its contributing options the same way (its `optionPool`
            // is a JS Map in insertion order, so it MUST sort to match). IEEE-754
            // addition is non-associative, so a divergent `total += per_option_sum`
            // order would make `measured` differ bit-for-bit and, via `tick → λ →
            // resolve`, flip a tie into the OPPOSITE queue. Do not change this to any
            // non-sorted iteration. Same discipline as `Tensor::dims_of`/`reduce_sum`.
            let mut total = 0.0;
            for (option, opt_pool) in &self.option_pool {
                if opt_pool != pool {
                    continue;
                }
                if let Ok(Some(change)) =
                    t.reduce_sum(from_level, to_level, target_scope, option, &dim)
                {
                    if change.next.is_finite() {
                        total += change.next;
                    }
                }
            }
            measured.insert(pool.clone(), total);
        }
        measured
    }
}

/// An **edge-local token-bucket admission gate** — the within-window rate limiter.
///
/// The slow price models a *cumulative* (daily) quota, which stays **slack** for a
/// purely *rate*-limited free tier — so it cannot pre-empt a mid-window `429`
/// (`ROUTING-MODEL.md` §6 "trap"). This bucket is the distinct, sub-second admission
/// gate that does: it holds `capacity` tokens, refills at `refill_per_sec`, and
/// **admits** a call only when a token is available. It is *local* (every edge runs
/// its own) and *non-route-time* — orthogonal to the `+∞` capability gate `resolve`
/// applies, and orthogonal to the slow steady-state price.
///
/// Time is injected (a monotonic seconds reading) so the two cores step it
/// identically — decision-equivalent, no wall-clock hidden state.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    tokens: f64,
    last: f64,
}

impl TokenBucket {
    /// A bucket of `capacity` tokens refilling at `refill_per_sec`, starting **full**
    /// at time `now` (seconds). A non-finite/negative capacity floors to `0` (admits
    /// nothing); a non-finite/negative refill floors to `0` (never refills) — both
    /// safe, never a non-finite leak.
    pub fn new(capacity: f64, refill_per_sec: f64, now: f64) -> Self {
        let capacity = if capacity.is_finite() && capacity > 0.0 {
            capacity
        } else {
            0.0
        };
        let refill_per_sec = if refill_per_sec.is_finite() && refill_per_sec > 0.0 {
            refill_per_sec
        } else {
            0.0
        };
        let last = if now.is_finite() { now } else { 0.0 };
        TokenBucket {
            capacity,
            refill_per_sec,
            tokens: capacity,
            last,
        }
    }

    /// Refill to the present `now`: add `refill_per_sec · Δt`, clamped to `capacity`.
    /// Time going backwards (or a non-finite `now`) advances nothing — never a refund.
    fn refill(&mut self, now: f64) {
        if !now.is_finite() || now <= self.last {
            if now.is_finite() {
                self.last = now.max(self.last);
            }
            return;
        }
        let dt = now - self.last;
        self.tokens = (self.tokens + dt * self.refill_per_sec).min(self.capacity);
        self.last = now;
    }

    /// **Admit one call** at time `now`: refill, then if a whole token is available
    /// consume it and return `true` (admit); otherwise return `false` (the caller must
    /// hold / spill to the next option). The sub-second gate the slow price cannot be.
    pub fn admit(&mut self, now: f64) -> bool {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// The tokens that WOULD be available at `now` (after a refill) — a pure read for
    /// inspection/tests. It does NOT mutate the bucket: probing `available(future)` must
    /// not advance the clock or top up tokens, or a later `admit(earlier)` would see time
    /// going backwards. Mirrors the non-mutating TS `available`.
    pub fn available(&self, now: f64) -> f64 {
        let dt = (now - self.last).max(0.0);
        (self.tokens + dt * self.refill_per_sec).min(self.capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{scope, Cursor, Scope, Tensor};

    #[test]
    fn price_is_a_shared_weight_not_the_gas_pedal() {
        // Two free options (financial 0), each drawing on its own pool. The price is
        // published as a SHARED weight on budget:<pool>; resolve adds λ·usage.
        let mut t = Tensor::new(["global"]);
        let root = Scope::new();
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", &root, "financial", 1.0).unwrap();
            w.set_value("global", &root, "free_a", "financial", 0.0)
                .unwrap();
            w.set_value("global", &root, "free_b", "financial", 0.0)
                .unwrap();
        }
        let budget = Budget::new()
            .with_pool("acct_a", 100.0)
            .with_pool("acct_b", 100.0)
            .map_option("free_a", "acct_a")
            .map_option("free_b", "acct_b");
        // Each option draws 1 unit per call onto its pool dim.
        {
            let mut w = t.writer("global").unwrap();
            budget
                .write_usage(&mut w, "global", &root, "free_a", 1.0)
                .unwrap();
            budget
                .write_usage(&mut w, "global", &root, "free_b", 1.0)
                .unwrap();
        }
        // Both idle ⇒ both prices 0 ⇒ tie broken by name.
        {
            let mut w = t.writer("global").unwrap();
            budget.publish_prices(&mut w, "global", &root).unwrap();
        }
        assert_eq!(t.resolve(&Cursor::new()), vec!["free_a", "free_b"]);

        // Now pool acct_a saturated: raise only its price. free_a's score climbs
        // (λ_a·usage added), free_b stays free ⇒ free_b sorts to the FRONT.
        let mut saturated = budget.clone();
        saturated.pools.get_mut("acct_a").unwrap().lambda = 5.0;
        {
            let mut w = t.writer("global").unwrap();
            saturated.publish_prices(&mut w, "global", &root).unwrap();
        }
        assert_eq!(t.resolve(&Cursor::new()), vec!["free_b", "free_a"]);
    }

    #[test]
    fn tick_raises_price_over_cap_and_decays_when_idle() {
        let mut b = Budget::new().with_step(0.1).with_pool("p", 100.0);
        // Over-consumed: usage 150 > cap 100 ⇒ λ rises by α·(150-100) = 0.1·50 = 5.
        let mut m = BTreeMap::new();
        m.insert("p".to_string(), 150.0);
        b.tick(&m);
        assert_eq!(b.lambda("p"), 5.0);
        // Still over: another +5 ⇒ 10.
        b.tick(&m);
        assert_eq!(b.lambda("p"), 10.0);
        // Idle window (usage 0 < cap 100): λ ← max(0, 10 + 0.1·(0-100)) = max(0,0) = 0.
        let idle = BTreeMap::new();
        b.tick(&idle);
        assert_eq!(b.lambda("p"), 0.0);
        // Projection floor: a deeply idle pool never goes negative.
        b.tick(&idle);
        assert_eq!(b.lambda("p"), 0.0);
    }

    #[test]
    fn saturation_simulation_load_sheds_to_the_idle_pool() {
        // Two free providers, two pools. We hammer pool "hot" past its cap each
        // window while "cold" stays idle. As the dual step raises the hot price its
        // option's score climbs and it sorts BEHIND the still-free cold option — load
        // sheds to the idle pool, with no `if 429 then rotate` branch anywhere.
        //
        // Faithful split of the two timescales (`ROUTING-MODEL.md` §6):
        //  - the PRICE/judgement tensor `t` is long-lived (its prices persist window
        //    to window);
        //  - the per-window USAGE feeder is a FRESH tensor each window — exactly the
        //    orchestrator's window accumulator, reset/pruned per window by
        //    live-scoping (the persistence port), so reduce_sum sums only THIS
        //    window's draws, never an ever-growing history.
        let mut t = Tensor::new(["global"]);
        let root = Scope::new();
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", &root, "financial", 1.0).unwrap();
            w.set_value("global", &root, "hot", "financial", 0.0)
                .unwrap();
            w.set_value("global", &root, "cold", "financial", 0.0)
                .unwrap();
            // Each option draws 1 unit on its pool per resolve (the steady-state probe).
            w.set_value("global", &root, "hot", &budget_dim("hot_pool"), 1.0)
                .unwrap();
            w.set_value("global", &root, "cold", &budget_dim("cold_pool"), 1.0)
                .unwrap();
        }
        let mut budget = Budget::new()
            .with_step(0.05)
            .with_pool("hot_pool", 10.0)
            .with_pool("cold_pool", 10.0)
            .map_option("hot", "hot_pool")
            .map_option("cold", "cold_pool");

        // Before any saturation: both free, both prices 0 ⇒ tie by name ⇒ cold, hot.
        {
            let mut w = t.writer("global").unwrap();
            budget.publish_prices(&mut w, "global", &root).unwrap();
        }
        assert_eq!(t.resolve(&Cursor::new()), vec!["cold", "hot"]);
        // hot is free and leads-or-ties — it WILL be picked, so it draws down its pool.

        // Simulate windows: each window 30 jobs hit "hot" (usage 30 ≫ cap 10), none
        // hit "cold". A fresh feeder per window → tick → publish → re-resolve.
        let mut order = Vec::new();
        for _win in 0..8 {
            let mut feeder = Tensor::new(["window", "job"]);
            for j in 0..30 {
                let inst = format!("j{j}");
                let job_scope = scope([("window", "agg"), ("job", inst.as_str())]);
                let mut w = feeder.writer("job").unwrap();
                budget
                    .write_usage(&mut w, "job", &job_scope, "hot", 1.0)
                    .unwrap();
            }
            let measured =
                budget.aggregate_usage(&mut feeder, "job", "window", &scope([("window", "agg")]));
            assert_eq!(measured.get("hot_pool").copied(), Some(30.0));
            assert_eq!(measured.get("cold_pool").copied(), Some(0.0));
            budget.tick(&measured);
            {
                let mut w = t.writer("global").unwrap();
                budget.publish_prices(&mut w, "global", &root).unwrap();
            }
            order = t.resolve(&Cursor::new());
        }
        // After saturation the hot pool's price has risen enough that the cold option
        // (still free, λ=0) leads and hot trails — load shed onto the idle pool.
        assert_eq!(order, vec!["cold", "hot"]);
        assert!(
            budget.lambda("hot_pool") > 0.0,
            "hot price should have risen"
        );
        assert_eq!(budget.lambda("cold_pool"), 0.0, "cold price stays at zero");
    }

    #[test]
    fn aggregate_usage_sums_draws_across_instances() {
        let mut t = Tensor::new(["global", "job"]);
        let budget = Budget::new()
            .with_pool("acct", 50.0)
            .map_option("m1", "acct")
            .map_option("m2", "acct");
        // Three jobs draw on the shared account via two options m1, m2.
        {
            let mut w = t.writer("job").unwrap();
            budget
                .write_usage(&mut w, "job", &scope([("job", "j1")]), "m1", 10.0)
                .unwrap();
            budget
                .write_usage(&mut w, "job", &scope([("job", "j2")]), "m1", 20.0)
                .unwrap();
            budget
                .write_usage(&mut w, "job", &scope([("job", "j3")]), "m2", 5.0)
                .unwrap();
        }
        let m = budget.aggregate_usage(&mut t, "job", "global", &Scope::new());
        // Pool total is the SUM of every draw on it: 10 + 20 + 5 = 35.
        assert_eq!(m.get("acct").copied(), Some(35.0));
    }

    #[test]
    fn unmapped_option_carries_no_budget_term() {
        let budget = Budget::new().with_pool("p", 10.0);
        let mut t = Tensor::new(["global"]);
        let mut w = t.writer("global").unwrap();
        // "lonely" maps to no pool ⇒ write_usage writes nothing.
        assert_eq!(
            budget.write_usage(&mut w, "global", &Scope::new(), "lonely", 9.0),
            Ok(false)
        );
    }

    #[test]
    fn token_bucket_admits_then_throttles_then_refills() {
        // Capacity 3, refill 1 token/sec. Burst of 3 admits, 4th denied; after 1s one
        // token is back ⇒ one more admit.
        let mut tb = TokenBucket::new(3.0, 1.0, 0.0);
        assert!(tb.admit(0.0)); // 3 → 2
        assert!(tb.admit(0.0)); // 2 → 1
        assert!(tb.admit(0.0)); // 1 → 0
        assert!(!tb.admit(0.0)); // empty → denied (the mid-window 429 the price can't stop)
        assert!(!tb.admit(0.5)); // half a token refilled, still < 1 → denied
        assert!(tb.admit(1.0)); // 1.0 token refilled → admitted
        assert!(!tb.admit(1.0)); // back to empty → denied
    }

    #[test]
    fn token_bucket_caps_refill_and_ignores_time_going_back() {
        let mut tb = TokenBucket::new(2.0, 5.0, 0.0);
        // Idle 100s would refill 500 tokens but capacity caps at 2.
        assert_eq!(tb.available(100.0), 2.0);
        // Time going backwards never refunds.
        assert!(tb.admit(100.0)); // 2 → 1
        assert_eq!(tb.available(50.0), 1.0); // backwards: no change
    }

    #[test]
    fn lambda_is_clamped_to_a_finite_ceiling_and_the_saturated_option_sorts_last() {
        // A runaway step on a pool far over cap drives λ to +∞ without the ceiling;
        // +∞ is SKIPPED by resolve so the saturated option would route FREE (front). With
        // the MAX_LAMBDA clamp λ stays finite and the saturated option sorts LAST. The step
        // is chosen so `next` genuinely OVERFLOWS to +∞ — making this test non-vacuous:
        // dropping the clamp flips the queue (verified), it does not merely re-store 1e300.
        let mut t = Tensor::new(["global"]);
        let root = Scope::new();
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", &root, "financial", 1.0).unwrap();
            w.set_value("global", &root, "saturated", "financial", 0.0)
                .unwrap();
            w.set_value("global", &root, "rival", "financial", 1.0)
                .unwrap();
            // saturated draws 1 unit on its pool each call.
            w.set_value("global", &root, "saturated", &budget_dim("p"), 1.0)
                .unwrap();
        }
        let mut b = Budget::new()
            .with_step(1e300) // absurd step …
            .with_pool("p", 0.0)
            .map_option("saturated", "p");
        // … on a pool over cap: next = 0 + 1e300·(1e10 - 0) = 1e310 → OVERFLOWS to +∞.
        let mut m = BTreeMap::new();
        m.insert("p".to_string(), 1e10);
        b.tick(&m);
        let lam = b.lambda("p");
        assert!(lam.is_finite(), "λ must stay finite");
        assert_eq!(lam, MAX_LAMBDA, "λ pins to the finite ceiling");
        {
            let mut w = t.writer("global").unwrap();
            b.publish_prices(&mut w, "global", &root).unwrap();
        }
        // saturated scores λ·1 = 1e300 ≫ rival's 1.0 → rival first, saturated LAST.
        assert_eq!(t.resolve(&Cursor::new()), vec!["rival", "saturated"]);
    }

    #[test]
    fn non_finite_usage_writes_nothing_and_keeps_the_option_routable() {
        // A non-finite usage must NOT land (it would read as a +∞ gate and drop the
        // option from routing). write_usage returns Ok(false) and resolve still keeps it;
        // aggregate_usage reports 0 for the pool (no draw recorded).
        let mut t = Tensor::new(["global", "job"]);
        let budget = Budget::new().with_pool("p", 100.0).map_option("m", "p");
        let job = scope([("job", "j1")]);
        {
            let mut w = t.writer("job").unwrap();
            w.set_weight("job", &Scope::new(), "financial", 1.0)
                .unwrap();
            w.set_value("job", &job, "m", "financial", 0.0).unwrap();
            // A NaN usage on the mapped option — must write nothing.
            assert_eq!(
                budget.write_usage(&mut w, "job", &job, "m", f64::NAN),
                Ok(false)
            );
            // An infinite usage likewise.
            assert_eq!(
                budget.write_usage(&mut w, "job", &job, "m", f64::INFINITY),
                Ok(false)
            );
        }
        // The option is still routable (no +∞ gate leaked from the usage).
        let mut at = Cursor::new();
        at.insert("job".into(), "j1".into());
        assert_eq!(t.resolve(&at), vec!["m"]);
        // No draw recorded → the pool aggregates to 0.
        let m = budget.aggregate_usage(&mut t, "job", "global", &Scope::new());
        assert_eq!(m.get("p").copied(), Some(0.0));
    }

    #[test]
    fn invalid_step_and_cap_are_floored() {
        let b = Budget::new().with_step(0.0).with_step(-1.0); // both rejected
        assert_eq!(b.step(), DEFAULT_STEP); // stays at the default
        let b2 = Budget::new()
            .with_pool("bad", f64::NAN)
            .with_pool("neg", -5.0);
        assert_eq!(b2.pool("bad").unwrap().cap, 0.0);
        assert_eq!(b2.pool("neg").unwrap().cap, 0.0);
    }
}
