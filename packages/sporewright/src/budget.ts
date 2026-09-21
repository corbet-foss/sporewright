// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * The **budget / price** layer — coupling constraints, priced as shared weights.
 * Decision-equivalent with the Rust `budget.rs`. See docs/MODEL.md §6 and
 * docs/ROUTING-MODEL.md §6.
 *
 * Some quantities are not judgements about one task; they are **shared, finite pools
 * that many tasks draw down together** (a provider account's daily quota, a target's
 * per-IP rate). The decision for task A depends on what B, C, D already consumed this
 * window — a *coupling* constraint that a per-task greedy choice cannot price in
 * isolation.
 *
 * The standard decoupling is **Lagrangian dual decomposition**: relax each budget
 * constraint into the objective with a **shadow price** `λ_pool ≥ 0`, the marginal
 * value of one more unit of the scarce pool. The elegant tensor form **needs no
 * engine change**:
 *
 *  - a pool's usage rides as the **value** on a per-task `budget:<pool>` dim
 *    (`usage_o`, this option's draw on the pool);
 *  - the pool's price `λ_pool` rides as the **shared weight** on that same dim
 *    ({@link Writer.setWeight} at `option=""`, inherited by every contender).
 *
 * So `resolve`'s Σ value·weight *already* contributes `λ_pool · usage_o` to every
 * option drawing on the pool — exactly the relaxed objective. As a pool saturates its
 * `λ_pool` rises, its contribution to every drawing option's score rises, and load
 * **self-rebalances** away from the scarce pool (Story 1). No rebalancing rule is
 * written; the shadow price does the job.
 *
 * **This is the additive price term, not the gas pedal.** The gas pedal
 * ({@link Writer.setOptionWeight}) is a *per-option, multiplicative* override that
 * scales one option's existing judgement score — a different mechanism. The price is
 * an *additive* `λ·usage` term shared across every option in the pool. Using the gas
 * pedal here would be wrong.
 *
 * Three pieces live here:
 *  1. {@link Budget} — the pools `{cap, lambda}` and the option→pool mapping, plus the
 *     `budget:<pool>` dim naming and helpers to write usage / publish prices.
 *  2. {@link Budget.tick} — the **single-writer** slow-tick price update
 *     `λ ← max(0, λ + α·(usage − cap))`, an **online projected subgradient** with a
 *     small *constant/floored* step (`[·]₊` enforces `λ ≥ 0`). Quotas refill, so
 *     there is no static optimum — we target **dynamic regret** with a constant step,
 *     never the textbook diminishing one. The accumulator is non-idempotent and the
 *     substrate is last-write-wins, so this is a centralised hub step, never
 *     distributed (devices only *read* prices).
 *  3. {@link TokenBucket} — a **separate, edge-local token-bucket admission gate**:
 *     the daily-quota price models a *cumulative* cap and stays slack for a purely
 *     *rate*-limited tier, so it cannot stop a mid-window `429`. The bucket is the
 *     sub-second within-window rate limiter that does.
 *
 * Usage is aggregated with {@link Tensor.reduceSum} (the additive feeder) — **not**
 * `reduceMedian`: total draw on a pool is a sum.
 */
import type { Scope, Tensor, WriteError, Writer } from "./tensor";

/** The budget-dim prefix: a pool's per-task usage and shared price live on
 *  `budget:<pool>`. The **value** is `usage_o` (this option's draw); the **shared
 *  weight** is `λ_pool` (the price). `resolve` then adds `λ_pool · usage_o` to every
 *  drawing option's score — the relaxed Lagrangian term, for free. Mirrors Rust. */
export const BUDGET_PREFIX = "budget:";

/** Compose the budget dimension `budget:<pool>` for a resource pool. */
export function budgetDim(pool: string): string {
  return `${BUDGET_PREFIX}${pool}`;
}

/** One resource pool: a shared finite cap and its current shadow price. */
export interface Pool {
  /** The window cap — the ceiling on summed usage before the price starts rising. */
  cap: number;
  /** The current shadow price `λ ≥ 0` (the budget dim's shared weight). Starts at `0`
   *  (an idle pool is free) and rises as usage exceeds the cap. */
  lambda: number;
}

/** The default constant step: small enough not to overshoot, floored so the price
 *  keeps reacting to a refilling quota. Override via {@link Budget.withStep}. */
export const DEFAULT_STEP = 0.1;

/** The smallest step permitted — a floor guaranteeing the price never freezes (the
 *  dynamic-regret guarantee: a constant, never-diminishing reaction). */
export const MIN_STEP = 1e-6;

/** The finite ceiling on the shadow price `λ`. A runaway/corrupt step could otherwise
 *  drive `λ` to `+∞`; since `resolve` SKIPS a non-finite weight, a `+∞`-priced
 *  (maximally-saturated) option would score as FREE and sort to the FRONT — the exact
 *  INVERSION of the intended shed. Clamping to `1e300` (finite, above every realistic
 *  price and above the `~1.1e16` the feeder golden pins) keeps a saturated option
 *  sorting LAST without ever becoming `+∞`. Mirrors Rust's `MAX_LAMBDA`. */
export const MAX_LAMBDA = 1e300;

/** The budget book: the pools `{cap, lambda}` keyed by pool id, and the option→pool
 *  mapping. A single hub owns this and advances the prices on the slow tick; devices
 *  read the published prices off the tensor and never mutate them. */
export class Budget {
  private readonly pools = new Map<string, Pool>();
  /** option → the pool it draws on. An option absent from the map draws on no pool:
   *  it carries no budget term and is priced on judgement alone. */
  private readonly optionPool = new Map<string, string>();
  /** The subgradient step `α > 0` — a small **constant** (dynamic-regret target),
   *  floored away from zero so the price always reacts. Never diminishing. */
  private stepVal = DEFAULT_STEP;

  /** Override the constant subgradient step `α`. A non-finite or below-floor step is
   *  rejected (the step stays at its prior value) — a `0`, negative, or NaN step would
   *  freeze or invert the price, the exact bug the dynamic-regret constant-step rule
   *  exists to prevent. Returns `this` for chaining. */
  withStep(step: number): this {
    if (Number.isFinite(step) && step >= MIN_STEP) this.stepVal = step;
    return this;
  }

  /** The effective step `α` (always finite and `≥ MIN_STEP`). */
  step(): number {
    return Number.isFinite(this.stepVal) && this.stepVal >= MIN_STEP ? this.stepVal : MIN_STEP;
  }

  /** Declare a pool with its window `cap`, price starting at `0` (idle ⇒ free). A
   *  non-finite or negative cap is clamped to `0` (a degenerate `0` cap prices any
   *  positive usage immediately, never a non-finite leak). Returns `this`. */
  withPool(pool: string, cap: number): this {
    const c = Number.isFinite(cap) && cap >= 0 ? cap : 0;
    this.pools.set(pool, { cap: c, lambda: 0 });
    return this;
  }

  /** Map an `option` onto the `pool` it draws from. Drawing on a pool means the option
   *  carries a `budget:<pool>` usage value and inherits that pool's price. Returns `this`. */
  mapOption(option: string, pool: string): this {
    this.optionPool.set(option, pool);
    return this;
  }

  /** The pool an `option` draws on, if any. */
  poolOf(option: string): string | undefined {
    return this.optionPool.get(option);
  }

  /** A pool's current `{cap, lambda}`, if declared. Returns a shallow COPY so a caller
   *  cannot mutate the live price — "the price changes only via tick" is a structural
   *  guarantee, not a convention (mirrors Rust's `pub(crate) lambda`). */
  pool(pool: string): Pool | undefined {
    const p = this.pools.get(pool);
    return p ? { ...p } : undefined;
  }

  /** The current shadow price `λ` for a pool (`0` if undeclared — an unknown pool is
   *  free, never a throw). */
  lambda(pool: string): number {
    return this.pools.get(pool)?.lambda ?? 0;
  }

  /** The pool ids, in insertion order. */
  poolIds(): string[] {
    return [...this.pools.keys()];
  }

  /** **Write a task's usage** for an option onto its pool's `budget:<pool>` dim, as a
   *  per-task **value** at `(level, inst, option)`. The summed draw on the pool is
   *  later rolled up with {@link Tensor.reduceSum}. An option that maps to no pool
   *  writes nothing.
   *
   *  Returns `false` when the option maps to no pool (nothing written), `true` when
   *  the usage landed, or throws nothing — a {@link WriteError} is returned via the
   *  out-param style of the core (so this returns `WriteError` on a write failure). */
  writeUsage(
    writer: Writer,
    layer: string,
    cellScope: Scope,
    option: string,
    usage: number,
  ): boolean | WriteError {
    const pool = this.poolOf(option);
    if (pool === undefined) return false;
    // Guard a non-finite usage: a `budget:<pool>` cell is a VALUE cell, and resolve
    // DROPS any option carrying a non-finite value (it reads as a +∞ gate). Writing a
    // NaN/Infinity usage would silently gate the option out instead of pricing its draw.
    // Write NOTHING (absence over a fabricated 0) — the option stays routable.
    if (!Number.isFinite(usage)) return false;
    const e = writer.setValue(layer, cellScope, option, budgetDim(pool), usage);
    if (e) return e;
    return true;
  }

  /** **Publish every pool's price** onto the tensor as the **shared weight** on its
   *  `budget:<pool>` dim (`option=""`), so `resolve` adds `λ_pool · usage_o` to every
   *  option drawing on the pool. This is the additive coupling term — written via
   *  {@link Writer.setWeight} (shared), **never** {@link Writer.setOptionWeight} (the
   *  per-option multiplicative gas pedal). Single-writer: only the hub calls this, on
   *  the slow tick, after {@link Budget.tick}. Returns the first {@link WriteError} or
   *  `undefined`. */
  publishPrices(writer: Writer, layer: string, cellScope: Scope): WriteError | undefined {
    for (const [pool, p] of this.pools) {
      // Defence-in-depth: never publish a non-finite price (a `+∞` weight is SKIPPED by
      // resolve and would route a saturated option FREE). `tick` already clamps to
      // MAX_LAMBDA, but a directly-set λ must be coerced too.
      const lambda = Number.isFinite(p.lambda) ? p.lambda : MAX_LAMBDA;
      const e = writer.setWeight(layer, cellScope, budgetDim(pool), lambda);
      if (e) return e;
    }
    return undefined;
  }

  /** **The slow-tick dual step (single-writer).** For each pool, read its summed usage
   *  this window from `measured` and nudge the price toward clearing the cap:
   *
   *      λ ← max(0, λ + α·(usage − cap))
   *
   *  an **online projected subgradient** with the constant step `α` ({@link Budget.step})
   *  and the `[·]₊` projection (`λ ≥ 0`). Over-consumed (`usage > cap`) ⇒ the price
   *  rises; idle (`usage < cap`) ⇒ it falls back toward `0`. Because quotas refill there
   *  is no static optimum — the constant step targets **dynamic regret**, never a
   *  diminishing schedule. A pool absent from `measured` is treated as `0` usage (idle ⇒
   *  decay). A non-finite measured usage is ignored (the pool's price holds) — a gate
   *  leak must never poison the accumulator. */
  tick(measured: Map<string, number>): void {
    const alpha = this.step();
    for (const [pool, p] of this.pools) {
      let usage: number;
      if (measured.has(pool)) {
        const u = measured.get(pool)!;
        if (!Number.isFinite(u)) continue; // non-finite measured usage → hold the price.
        usage = u;
      } else {
        usage = 0; // unmeasured ⇒ idle this window.
      }
      const next = p.lambda + alpha * (usage - p.cap);
      // Clamp the new price to a FINITE ceiling: `[·]₊` keeps λ ≥ 0, MAX_LAMBDA keeps it
      // from ever reaching `+∞` (a `+∞` weight is skipped by resolve and would route a
      // saturated option FREE — the opposite of the intended shed).
      p.lambda = Number.isFinite(next) && next > 0 ? Math.min(next, MAX_LAMBDA) : next > 0 ? MAX_LAMBDA : 0;
    }
  }

  /** **Aggregate this window's usage** from the tensor: for each pool, sum the per-task
   *  `budget:<pool>` usage values across every instance at `fromLevel` via
   *  {@link Tensor.reduceSum} (which returns each option's summed draw, also writing it
   *  at `toLevel`/`toInst`), accumulated into the `measured` map {@link Budget.tick} consumes.
   *
   *  `reduceSum` is the **additive** feeder: a budget price needs the *sum* of draws,
   *  not the median. A pool with no usage this window reports `0` (idle), so its price
   *  decays on the next tick. */
  aggregateUsage(t: Tensor, fromLayer: string, toLayer: string, targetScope: Scope): Map<string, number> {
    const measured = new Map<string, number>();
    for (const pool of this.pools.keys()) {
      const dim = budgetDim(pool);
      // The per-option pool totals must be summed in a cross-core-canonical order.
      // Rust iterates `option_pool` (a BTreeMap, SORTED by option name); TS's
      // `optionPool` is a JS Map in INSERTION order. With 3+ options of disparate
      // magnitude on one pool, `total += per_option_sum` accumulates in a different
      // order on each core — and IEEE-754 addition is non-associative, so `measured`
      // would diverge bit-for-bit, then `tick → λ → resolve` could flip a tie and
      // produce the OPPOSITE queue. So we sort the contributing options by name first,
      // mirroring Rust's BTreeMap. Same discipline as `Tensor.dimsOf`/`reduceSum`.
      const opts: string[] = [];
      for (const [option, optPool] of this.optionPool) {
        if (optPool === pool) opts.push(option);
      }
      opts.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      let total = 0;
      for (const option of opts) {
        const change = t.reduceSum(fromLayer, toLayer, targetScope, option, dim);
        if (change !== undefined && typeof change !== "string" && Number.isFinite(change.next)) total += change.next;
      }
      measured.set(pool, total);
    }
    return measured;
  }
}

/** An **edge-local token-bucket admission gate** — the within-window rate limiter.
 *
 *  The slow price models a *cumulative* (daily) quota, which stays **slack** for a
 *  purely *rate*-limited free tier — so it cannot pre-empt a mid-window `429`. This
 *  bucket is the distinct, sub-second admission gate that does: it holds `capacity`
 *  tokens, refills at `refillPerSec`, and **admits** a call only when a token is
 *  available. It is *local* (every edge runs its own) and *non-route-time* —
 *  orthogonal to the `+∞` capability gate `resolve` applies, and orthogonal to the
 *  slow steady-state price.
 *
 *  Time is injected (a monotonic seconds reading) so the two cores step it identically
 *  — decision-equivalent, no wall-clock hidden state. */
export class TokenBucket {
  private readonly capacity: number;
  private readonly refillPerSec: number;
  private tokens: number;
  private last: number;

  /** A bucket of `capacity` tokens refilling at `refillPerSec`, starting **full** at
   *  time `now` (seconds). A non-finite/negative capacity floors to `0` (admits
   *  nothing); a non-finite/negative refill floors to `0` (never refills) — both safe,
   *  never a non-finite leak. */
  constructor(capacity: number, refillPerSec: number, now: number) {
    this.capacity = Number.isFinite(capacity) && capacity > 0 ? capacity : 0;
    this.refillPerSec = Number.isFinite(refillPerSec) && refillPerSec > 0 ? refillPerSec : 0;
    this.last = Number.isFinite(now) ? now : 0;
    this.tokens = this.capacity;
  }

  /** Refill to the present `now`: add `refillPerSec · Δt`, clamped to `capacity`. Time
   *  going backwards (or a non-finite `now`) advances nothing — never a refund. */
  private refill(now: number): void {
    if (!Number.isFinite(now) || now <= this.last) {
      if (Number.isFinite(now)) this.last = Math.max(now, this.last);
      return;
    }
    const dt = now - this.last;
    this.tokens = Math.min(this.tokens + dt * this.refillPerSec, this.capacity);
    this.last = now;
  }

  /** **Admit one call** at time `now`: refill, then if a whole token is available
   *  consume it and return `true` (admit); otherwise return `false` (the caller must
   *  hold / spill to the next option). The sub-second gate the slow price cannot be. */
  admit(now: number): boolean {
    this.refill(now);
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }

  /** The tokens that WOULD be available at `now` (after a refill) — a pure read for
   *  inspection/tests. It does NOT mutate the bucket: probing `available(future)` must
   *  not advance the clock or top up tokens, or a later `admit(earlier)` would see time
   *  going backwards. Mirrors the non-mutating Rust `available`. */
  available(now: number): number {
    const dt = Math.max(now - this.last, 0);
    return Math.min(this.tokens + dt * this.refillPerSec, this.capacity);
  }
}
