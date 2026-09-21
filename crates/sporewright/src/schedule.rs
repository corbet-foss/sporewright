// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Generic work-scheduling model — product-agnostic.
//!
//! How to PRIORITISE a unit of work (urgency from recency × trust × volatility),
//! BUCKET a continuous priority into a few queue levels, apply the LEASE lifecycle
//! (in-flight exclusion + TTL redelivery), and SIZE a work budget from a saturation
//! signal (invest when idle, back off when saturated).
//!
//! This is pure model — the same shape whether the work is re-scraping a job
//! posting or re-running an LLM call. The I/O that feeds it (DB queries, message
//! queues, HTTP) lives in the product server, never here.

/// Urgency of one unit of work. Recency (`staleness_secs`) is the base; a low-trust
/// last observer multiplies it up (corroborate sooner); provider-expected
/// `volatility` multiplies it up (work that changes/expires fast is re-checked
/// sooner). `observer_trust` is clamped to `[0,1]` (NaN reads as the `0.5` trust
/// default — `clamp` panics on NaN, so it never reaches it); pass `volatility = 1.0`
/// when unknown. A negative staleness or volatility reads as `0` (nothing overdue,
/// no change expected); a NaN staleness/volatility is likewise neutralised, so
/// this never panics. `±∞` inputs take the clamp's edge (`+∞` trust → full trust).
pub fn priority(staleness_secs: f64, observer_trust: f64, volatility: f64) -> f64 {
    let trust = if observer_trust.is_nan() {
        0.5 // unknown trust → the default: half urgency uplift, never a clamp panic.
    } else {
        observer_trust.clamp(0.0, 1.0)
    };
    let trust_deficit = (1.0 - trust).max(0.0);
    staleness_secs.max(0.0) * (1.0 + trust_deficit) * volatility.max(0.0)
}

/// Map a unit's rank within a published batch (0 = highest priority) to a queue
/// priority bucket in `1..=levels`. Message queues cap their priority levels, so a
/// continuous priority is bucketed; exact intra-batch order does not matter.
pub fn priority_bucket(rank: usize, total: usize, levels: u8) -> u8 {
    if levels == 0 {
        return 1;
    }
    if total <= 1 {
        return levels;
    }
    let frac = rank as f64 / (total - 1) as f64; // 0.0 (top) .. 1.0 (bottom)
    let bucket = ((1.0 - frac) * (levels as f64 - 1.0)).round() as i64 + 1;
    bucket.clamp(1, levels as i64) as u8
}

/// Default lease window: a unit handed to a worker is excluded from re-handing for
/// this long; if no result lands by then it ages back out = redelivery.
pub const DEFAULT_LEASE_TTL_SECS: u64 = 600;

/// Is a unit still in flight (leased and within its TTL)? `leased_at_secs = None`
/// means never leased = available. Epoch seconds. Redelivery is simply this going
/// false again — the lease state lives in the store, not in the queue.
pub fn lease_in_flight(leased_at_secs: Option<f64>, now_secs: f64, ttl_secs: f64) -> bool {
    match leased_at_secs {
        Some(t) => now_secs - t < ttl_secs,
        None => false,
    }
}

/// Fold a saturation signal into a work-budget envelope. `saturation ∈ [0,1]`
/// (0 = idle → invest at `ceiling`; 1 = saturated → back off to `floor`); linear
/// between. Never below 1. NaN is no signal → treated as idle (`0`, invest) rather
/// than panicking in `clamp`; `±∞` take the clamp's edge (`+∞` → saturated).
pub fn budget_from_saturation(saturation: f64, floor: u64, ceiling: u64) -> u64 {
    let (lo, hi) = if ceiling >= floor {
        (floor, ceiling)
    } else {
        (ceiling, floor)
    };
    let saturation = if saturation.is_nan() {
        0.0
    } else {
        saturation.clamp(0.0, 1.0)
    };
    let invest = (1.0 - saturation).max(0.0);
    (lo + ((hi - lo) as f64 * invest).round() as u64).max(1)
}

/// Saturation from a backlog against a cap: 0 at empty, rising to 1 at/over `cap`.
/// `cap = 0` yields 0 (no signal → treat as idle → invest). Integer inputs, so no
/// NaN path: the ratio is always finite and `.min(1.0)` only clamps the top.
pub fn backlog_saturation(backlog: u64, cap: u64) -> f64 {
    if cap == 0 {
        0.0
    } else {
        (backlog as f64 / cap as f64).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_rises_with_staleness_trust_deficit_and_volatility() {
        assert!(priority(10_000.0, 1.0, 1.0) > priority(1_000.0, 1.0, 1.0));
        // low trust -> deficit 1 -> ×2
        assert!((priority(1_000.0, 0.0, 1.0) - 2.0 * priority(1_000.0, 1.0, 1.0)).abs() < 1e-9);
        // volatility scales linearly
        assert!((priority(1_000.0, 1.0, 2.0) - 2.0 * priority(1_000.0, 1.0, 1.0)).abs() < 1e-9);
        assert_eq!(priority(0.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn buckets_map_top_to_bottom() {
        assert_eq!(priority_bucket(0, 10, 10), 10);
        assert_eq!(priority_bucket(9, 10, 10), 1);
        assert_eq!(priority_bucket(0, 1, 10), 10);
        assert!((1..=10).contains(&priority_bucket(5, 10, 10)));
    }

    #[test]
    fn lease_in_flight_respects_ttl() {
        assert!(!lease_in_flight(None, 100.0, 600.0)); // never leased
        assert!(lease_in_flight(Some(100.0), 200.0, 600.0)); // 100s < 600
        assert!(!lease_in_flight(Some(100.0), 800.0, 600.0)); // 700s >= 600 -> aged out
    }

    #[test]
    fn budget_invests_idle_backs_off_saturated() {
        assert_eq!(budget_from_saturation(0.0, 10, 100), 100); // idle -> ceiling
        assert_eq!(budget_from_saturation(1.0, 10, 100), 10); // saturated -> floor
        assert_eq!(budget_from_saturation(0.5, 10, 100), 55); // mid
        assert_eq!(backlog_saturation(0, 2000), 0.0);
        assert_eq!(backlog_saturation(1000, 2000), 0.5);
        assert_eq!(backlog_saturation(5000, 2000), 1.0); // clamped
    }

    #[test]
    fn non_finite_and_negative_inputs_never_panic_and_read_as_no_signal() {
        // f64::clamp panics on NaN: a NaN trust/saturation must NOT reach it.
        // NaN trust → the 0.5 default (deficit 0.5 → ×1.5 uplift); ±∞ take the
        // clamp's edge (+∞ → full trust → ×1.0; -∞ → zero trust → ×2.0).
        assert_eq!(priority(1_000.0, f64::NAN, 1.0), 1_500.0);
        assert_eq!(priority(1_000.0, f64::INFINITY, 1.0), 1_000.0);
        assert_eq!(priority(1_000.0, f64::NEG_INFINITY, 1.0), 2_000.0);
        // NaN saturation → idle → invest at the ceiling.
        assert_eq!(budget_from_saturation(f64::NAN, 10, 100), 100);
        assert_eq!(budget_from_saturation(f64::INFINITY, 10, 100), 10); // clamps to 1
        assert_eq!(budget_from_saturation(f64::NEG_INFINITY, 10, 100), 100); // clamps to 0
                                                                             // Negative staleness/volatility read as 0 (nothing overdue, no change).
        assert_eq!(priority(-5.0, 1.0, 1.0), 0.0);
        assert_eq!(priority(1_000.0, 1.0, -2.0), 0.0);
        // NaN staleness/volatility neutralise via max (NaN.max(0.0) == 0.0).
        assert_eq!(priority(f64::NAN, 1.0, 1.0), 0.0);
        assert_eq!(priority(1_000.0, 1.0, f64::NAN), 0.0);
        // cap = 0 → no signal → idle.
        assert_eq!(backlog_saturation(7, 0), 0.0);
        // Swapped floor/ceiling still envelopes; envelope never drops below 1.
        assert_eq!(budget_from_saturation(0.0, 100, 10), 100);
        assert_eq!(budget_from_saturation(1.0, 100, 10), 10);
        assert_eq!(budget_from_saturation(0.0, 0, 0), 1);
    }
}
