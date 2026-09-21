// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Per-dimension **normalization** — making `resolve`'s weighted sum honest.
 * Decision-equivalent with the Rust `normalize.rs`. See docs/MODEL.md §3 and
 * docs/ROUTING-MODEL.md §3 / §6.
 *
 * `resolve` is Σ value·weight over a *heterogeneous* dimension vector: `financial`
 * is dollars, `latency` is seconds, `quality` is a unit-gap. Summed raw, the
 * dimension with the largest native magnitude dominates and the weights stop being
 * honest **marginal rates of substitution** — at the scrape router's seeded weights
 * `financial` outweighs `latency` ~200×, so the order is effectively *lexicographic*
 * by cost rather than a real multi-objective trade-off.
 *
 * The fix is a **numéraire per dimension**: divide each cost by a per-dim scale so
 * costs are commensurable, and *then* the weight is a true MRS. The identity that
 * keeps the engine pure:
 *
 *     (value / scale) · weight  ===  value · (weight / scale)
 *
 * so normalization is a **write-side weight pre-divisor** — `resolve` itself never
 * changes (still Σ value·weight). A consumer that wants honest balanced scoring
 * **opts in** by deriving its effective weights through this helper before writing
 * them; a consumer happy with raw, possibly-lexicographic behaviour (the scrape
 * router preferring free) simply does not call it. The tensor stays a pure
 * Σ value·weight lens; the normalizer is thin consumer-side config — the SCOPE
 * split (mechanism here, the *scales* are product data).
 *
 * Two ways to use it, both pure, no tensor state:
 *  - {@link Norm.effectiveWeight} — the weight-side route (recommended): keep raw
 *    measured costs in the cells, fold `1/scale` into the weight you write.
 *  - {@link Norm.normalizeValue} — the value-side route ("normalise to [0,1] at
 *    seed time"): map a raw value into [0,1] before writing it.
 *
 * **Scope — judgement dims only (role A).** Normalization applies ONLY to *judgement*
 * dims (ROUTING-MODEL.md §3 role A: `financial`/`latency`/`quality`/`reliability`/
 * `freshness`). NEVER derive a scale for a `budget:<pool>` dim (role B): there the
 * **weight** is the shadow price `λ` and the **value** is the additive usage, so
 * `resolve` contributes the Lagrangian term `λ · usage`. Rescaling either side corrupts
 * it — `λ · (usage/scale)` or `(λ/scale) · usage` — so `λ` stops meaning "the marginal
 * value of one pool unit" and the dual step runs a scaled price against a raw cap. Gate
 * dims (role C, the `±∞` values) are already passed through untouched and must never
 * receive a scale either.
 */

/** A per-dimension scale map — the numéraire for each cost dimension. A dimension
 *  absent (or with a non-positive / non-finite scale) is identity (`scale = 1.0`),
 *  the safe default that makes opting-in additive and a partial spec harmless. */
export class Norm {
  private readonly scales = new Map<string, number>();

  /** Set the scale (numéraire) for a dimension. A non-finite or non-positive scale
   *  is rejected (the dimension stays identity) — a `0` or `±∞` scale would inject a
   *  non-finite or annihilating weight and silently corrupt the queue, the exact
   *  bug class normalization exists to prevent. Returns `this` for chaining. */
  withScale(dim: string, scale: number): this {
    if (Number.isFinite(scale) && scale > 0) this.scales.set(dim, scale);
    return this;
  }

  /** The effective scale for a dimension (`1.0` when unset / invalid). */
  scale(dim: string): number {
    const s = this.scales.get(dim);
    return s !== undefined && Number.isFinite(s) && s > 0 ? s : 1.0;
  }

  /** The **honest weight** to write for `dim` given the raw MRS weight: the raw
   *  weight divided by the dimension's scale. Writing this (via `setWeight`) makes
   *  Σ value·effectiveWeight equal Σ (value/scale)·rawWeight — the weighted sum is
   *  now over commensurable, normalized costs, so the raw weight is a true marginal
   *  rate of substitution. The engine is untouched: it still computes a plain Σ v·w. */
  effectiveWeight(dim: string, rawWeight: number): number {
    return rawWeight / this.scale(dim);
  }

  /** The **value-side** alternative: map a raw value into [0,1] by
   *  `(v - lo)/(hi - lo)` (min-max), clamped to [0,1]. For seeding already-
   *  commensurable values. A degenerate or non-finite range (`hi <= lo`, or either
   *  bound non-finite) returns the value unchanged — never a divide-by-zero, never a
   *  non-finite leak. A non-finite `v` (a gate) passes through untouched so gates
   *  still gate. */
  static normalizeValue(v: number, lo: number, hi: number): number {
    if (!Number.isFinite(v)) return v; // a gate (±∞/NaN) must stay a gate.
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return v; // degenerate → identity.
    const n = (v - lo) / (hi - lo);
    // Overflow path: a finite range that itself overflows (e.g. v=1e308, lo=-1e308,
    // hi=1e308) makes (v-lo)/(hi-lo) = Infinity/Infinity = NaN. Fall back to identity so
    // the "never a non-finite leak" guarantee holds unconditionally.
    if (!Number.isFinite(n)) return v;
    return n < 0 ? 0 : n > 1 ? 1 : n;
  }
}

/** Derive a **max-abs scale** from observed values: the largest absolute finite
 *  value (non-finite gates ignored). Dividing by it puts the dimension's costs in
 *  [-1, 1]. Returns `1.0` when there is no positive finite magnitude (all-zero /
 *  empty / all-gate), so the result is always a valid, queue-safe scale. */
export function maxAbsScale(values: Iterable<number>): number {
  let m = 0;
  for (const v of values) {
    if (Number.isFinite(v)) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
  }
  return Number.isFinite(m) && m > 0 ? m : 1.0;
}
