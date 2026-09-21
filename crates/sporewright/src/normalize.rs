// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Per-dimension **normalization** — making `resolve`'s weighted sum honest.
//!
//! `resolve` is `Σ value·weight` over a *heterogeneous* dimension vector
//! (`MODEL.md` §3, `ROUTING-MODEL.md` §3): `financial` is dollars, `latency` is
//! seconds, `quality` is a unit-gap. Summed raw, the dimension with the largest
//! native magnitude dominates and the weights stop being honest **marginal rates
//! of substitution** — at the scrape router's seeded weights `financial` outweighs
//! `latency` ~200×, so the order is effectively *lexicographic* by cost, not a real
//! multi-objective trade-off (`ROUTING-MODEL.md` §6, "Live vs. design").
//!
//! The fix is a **numéraire per dimension**: divide each cost by a per-dim scale so
//! all costs live on a common range, and *then* the weight is a true MRS. The key
//! identity keeps the engine pure:
//!
//! ```text
//!   (value / scale) · weight  ==  value · (weight / scale)
//! ```
//!
//! so normalization is a **write-side weight pre-divisor** — `resolve` itself never
//! changes (it is still `Σ value·weight`). A consumer that wants honest, balanced
//! multi-objective scoring **opts in** by deriving its effective weights through
//! this helper before writing them; a consumer that is happy with raw, possibly
//! lexicographic behaviour (the scrape router preferring free) simply does not call
//! it. The tensor stays a pure `Σ value·weight` lens; the normalizer is thin
//! consumer-side config — exactly the SCOPE split (mechanism here, the *scales*
//! are product data).
//!
//! Two ways to use it, both pure functions, no tensor state:
//!
//! * [`Norm::effective_weight`] — the **weight-side** route (recommended): keep the
//!   raw measured costs in the cells, fold `1/scale` into the weight you write. One
//!   write per dim, the values stay human-readable, and a re-measured cost still
//!   re-orders the queue correctly because the scale is in the weight.
//! * [`Norm::normalize_value`] — the **value-side** route ("normalise to `[0, 1]`
//!   at seed time"): map a raw value into `[0, 1]` by `(v - lo)/(hi - lo)` before
//!   writing it, when the consumer would rather store already-commensurable values.
//!
//! Both are decision-equivalent across the Rust and TS cores.
//!
//! **Scope — judgement dims only (role A).** Normalization applies ONLY to *judgement*
//! dims (`ROUTING-MODEL.md` §3 role A: `financial`/`latency`/`quality`/`reliability`/
//! `freshness`). NEVER derive a scale for a `budget:<pool>` dim (role B): there the
//! **weight** is the shadow price `λ` and the **value** is the additive usage, so
//! `resolve` contributes the Lagrangian term `λ · usage`. Rescaling either side corrupts
//! it — `λ · (usage/scale)` or `(λ/scale) · usage` — so `λ` stops meaning "the marginal
//! value of one pool unit" and the dual step runs a scaled price against a raw cap. Gate
//! dims (role C, the `±∞` values) are already passed through untouched and must never
//! receive a scale either.

use std::collections::BTreeMap;

/// A per-dimension scale map — the numéraire for each cost dimension. A dimension
/// absent from the map (or with a non-positive / non-finite scale) is treated as
/// scale `1.0`: *unnormalized*, the safe identity. This is what makes opting-in
/// additive and a partial spec harmless.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Norm {
    scales: BTreeMap<String, f64>,
}

impl Norm {
    /// An empty normalizer — every dimension is identity (`scale = 1.0`).
    pub fn new() -> Self {
        Norm {
            scales: BTreeMap::new(),
        }
    }

    /// Set the scale (numéraire) for a dimension. A non-finite or non-positive
    /// scale is rejected (the dimension stays identity) — a `0` or `±∞` scale would
    /// otherwise inject a non-finite or annihilating weight and silently corrupt the
    /// queue, which is exactly the class of bug normalization exists to prevent.
    pub fn with_scale(mut self, dim: &str, scale: f64) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.scales.insert(dim.to_string(), scale);
        }
        self
    }

    /// The effective scale for a dimension (`1.0` when unset / invalid).
    pub fn scale(&self, dim: &str) -> f64 {
        match self.scales.get(dim) {
            Some(s) if s.is_finite() && *s > 0.0 => *s,
            _ => 1.0,
        }
    }

    /// The **honest weight** to write for `dim` given the raw MRS weight: the raw
    /// weight divided by the dimension's scale. Writing this (via
    /// `Writer::set_weight`) makes `Σ value·effective_weight` equal to
    /// `Σ (value/scale)·raw_weight` — i.e. the weighted sum is now over
    /// commensurable, normalized costs, so the raw weight is a true marginal rate of
    /// substitution. The engine is untouched: it still computes a plain `Σ v·w`.
    pub fn effective_weight(&self, dim: &str, raw_weight: f64) -> f64 {
        raw_weight / self.scale(dim)
    }

    /// The **value-side** alternative: map a raw value into `[0, 1]` by
    /// `(v - lo)/(hi - lo)` (a min-max normalizer), clamped to `[0, 1]`. For
    /// seeding already-commensurable values when the consumer prefers normalized
    /// *values* over a normalized *weight*. A degenerate or non-finite range
    /// (`hi <= lo`, or either bound non-finite) returns the value unchanged — never
    /// a divide-by-zero, never a non-finite leak into a cell. A non-finite `v` (a
    /// gate) is passed through untouched so gates still gate.
    pub fn normalize_value(v: f64, lo: f64, hi: f64) -> f64 {
        if !v.is_finite() {
            return v; // a gate (±∞/NaN) must stay a gate — never normalized away.
        }
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return v; // degenerate range → identity, no divide-by-zero.
        }
        let n = (v - lo) / (hi - lo);
        if !n.is_finite() {
            // Overflow path: a finite range that itself overflows f64 (e.g. v=1e308,
            // lo=-1e308, hi=1e308) makes (v-lo)/(hi-lo) = inf/inf = NaN, which clamp
            // would pass through — a non-finite leak. Fall back to identity so the
            // docstring's "never a non-finite leak into a cell" holds unconditionally.
            return v;
        }
        n.clamp(0.0, 1.0)
    }
}

/// Derive a **max-abs scale** from a set of observed values: the largest absolute
/// finite value (non-finite gates ignored). Dividing by it puts the dimension's
/// costs in `[-1, 1]`. Returns `1.0` when there is no positive finite magnitude
/// (an all-zero / empty / all-gate dimension), so the result is always a valid,
/// queue-safe scale.
pub fn max_abs_scale(values: impl IntoIterator<Item = f64>) -> f64 {
    let m = values
        .into_iter()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if m.is_finite() && m > 0.0 {
        m
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Cursor, Scope, Tensor};

    #[test]
    fn effective_weight_is_raw_over_scale() {
        let n = Norm::new()
            .with_scale("financial", 0.01)
            .with_scale("latency", 5.0);
        // financial spans ~0.01 $/call, latency ~5 s; equal MRS weight 1.0 each.
        assert_eq!(n.effective_weight("financial", 1.0), 100.0);
        assert_eq!(n.effective_weight("latency", 1.0), 0.2);
        // an unset dim is identity.
        assert_eq!(n.effective_weight("quality", 3.0), 3.0);
    }

    #[test]
    fn invalid_scales_fall_back_to_identity() {
        let n = Norm::new()
            .with_scale("a", 0.0) // rejected (non-positive)
            .with_scale("b", f64::INFINITY) // rejected (non-finite)
            .with_scale("c", -2.0) // rejected (non-positive)
            .with_scale("d", 4.0); // accepted
        assert_eq!(n.scale("a"), 1.0);
        assert_eq!(n.scale("b"), 1.0);
        assert_eq!(n.scale("c"), 1.0);
        assert_eq!(n.scale("d"), 4.0);
    }

    #[test]
    fn normalize_value_min_max_and_degenerate_ranges() {
        assert_eq!(Norm::normalize_value(5.0, 0.0, 10.0), 0.5);
        assert_eq!(Norm::normalize_value(0.0, 0.0, 10.0), 0.0);
        assert_eq!(Norm::normalize_value(10.0, 0.0, 10.0), 1.0);
        // clamp out-of-range.
        assert_eq!(Norm::normalize_value(20.0, 0.0, 10.0), 1.0);
        assert_eq!(Norm::normalize_value(-5.0, 0.0, 10.0), 0.0);
        // degenerate range → identity (no divide-by-zero).
        assert_eq!(Norm::normalize_value(7.0, 3.0, 3.0), 7.0);
        assert_eq!(Norm::normalize_value(7.0, 5.0, 1.0), 7.0);
        // a gate passes through untouched.
        assert_eq!(
            Norm::normalize_value(f64::INFINITY, 0.0, 10.0),
            f64::INFINITY
        );
        // OVERFLOW: a finite range that overflows f64 makes (v-lo)/(hi-lo) = inf/inf =
        // NaN; we must fall back to identity, never leak a non-finite into a cell.
        let n = Norm::normalize_value(1e308, -1e308, 1e308);
        assert!(n.is_finite(), "overflow must not leak a non-finite value");
        assert_eq!(n, 1e308); // identity fallback
    }

    #[test]
    fn max_abs_scale_picks_the_largest_magnitude() {
        assert_eq!(max_abs_scale([0.2, -5.0, 3.0]), 5.0);
        assert_eq!(max_abs_scale([0.0, 0.0]), 1.0); // no magnitude → identity
        assert_eq!(max_abs_scale([f64::INFINITY, 2.0]), 2.0); // gate ignored
        assert_eq!(max_abs_scale(std::iter::empty()), 1.0);
    }

    #[test]
    fn normalization_flips_a_lexicographic_order_into_a_balanced_one() {
        // Two options, two competing dims. financial lives on ~0.01, latency on ~5.
        // a: cheap but slow; b: pricey but fast. With EQUAL MRS weights the honest
        // trade-off should hinge on normalized costs, not on which dim is "bigger".
        let mut t = Tensor::new(["global"]);
        let root = Scope::new();
        {
            let mut w = t.writer("global").unwrap();
            w.set_value("global", &root, "a", "financial", 0.002)
                .unwrap();
            w.set_value("global", &root, "a", "latency", 5.0).unwrap();
            w.set_value("global", &root, "b", "financial", 0.010)
                .unwrap();
            w.set_value("global", &root, "b", "latency", 0.5).unwrap();
        }

        // RAW, equal weights: latency's magnitude (~5) dwarfs financial (~0.01), so
        // the order is dominated by latency alone — `b` (fast) wins lexicographically.
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", &root, "financial", 1.0).unwrap();
            w.set_weight("global", &root, "latency", 1.0).unwrap();
        }
        assert_eq!(t.resolve(&Cursor::new()), vec!["b", "a"]);

        // NORMALIZED: scale each dim by its span, keep equal MRS weights. Now the
        // costs are commensurable in [0,1]-ish and the trade-off is honest.
        // a: 0.002/0.01 + 5/5   = 0.2 + 1.0 = 1.2
        // b: 0.010/0.01 + 0.5/5 = 1.0 + 0.1 = 1.1  → b still leads but only just,
        // and `a` is no longer crushed: the gap collapsed from latency-only to a
        // real two-objective sum.
        let n = Norm::new()
            .with_scale("financial", 0.01)
            .with_scale("latency", 5.0);
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight(
                "global",
                &root,
                "financial",
                n.effective_weight("financial", 1.0),
            )
            .unwrap();
            w.set_weight(
                "global",
                &root,
                "latency",
                n.effective_weight("latency", 1.0),
            )
            .unwrap();
        }
        assert_eq!(t.resolve(&Cursor::new()), vec!["b", "a"]);

        // And crank financial's MRS weight up: now the cheap option `a` overtakes —
        // which it could NEVER do under the raw, latency-dominated sum at any
        // reasonable financial weight. This is the Pareto-tilt the model promises.
        let n2 = n.clone();
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight(
                "global",
                &root,
                "financial",
                n2.effective_weight("financial", 3.0),
            )
            .unwrap();
        }
        // a: 0.002/0.01·3 + 5/5·1   = 0.6 + 1.0 = 1.6
        // b: 0.010/0.01·3 + 0.5/5·1 = 3.0 + 0.1 = 3.1  → a now wins on cost.
        assert_eq!(t.resolve(&Cursor::new()), vec!["a", "b"]);
    }
}
