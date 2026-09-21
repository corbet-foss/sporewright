// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Pure trust, freshness, and churn arithmetic.
//!
//! The host owns clocks, persistence, identities, and evidence collection. This
//! module only transforms explicit inputs, which keeps it useful in centralized
//! orchestrators and peer-local routers alike.

pub const TRUST_AGREE_BASE_GAIN: f64 = 0.02;
pub const TRUST_DISAGREE_BASE_LOSS: f64 = 0.04;
pub const TRUST_TIME_HALF_LIFE_HOURS: f64 = 48.0;
pub const TRUST_DEFAULT: f64 = 0.5;
pub const TRUST_TRUSTED_AT: f64 = 0.8;
pub const TRUST_LIMITED_BELOW: f64 = 0.3;
pub const VERIFY_CONFLICT_MIN_WEIGHT: f64 = 0.1;
pub const FACT_CHURN_COOLDOWN_THRESHOLD: u64 = 6;
pub const FACT_CHURN_WINDOW_MS: i64 = 86_400_000;
pub const FACT_COOLDOWN_MS: i64 = 86_400_000;

/// Clamp to `[0, 1]`. Non-finite evidence resets to the neutral default.
pub fn clamp01(value: f64) -> f64 {
    if !value.is_finite() {
        return TRUST_DEFAULT;
    }
    value.clamp(0.0, 1.0)
}

/// Exponential freshness weight with a 48-hour half-life.
pub fn time_weight(delta_hours: f64) -> f64 {
    if !delta_hours.is_finite() {
        return TRUST_DEFAULT;
    }
    2_f64.powf(-delta_hours.abs() / TRUST_TIME_HALF_LIFE_HOURS)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairTrustResult {
    pub incoming_trust: f64,
    pub current_trust: f64,
}

/// Symmetric pairwise trust update.
///
/// `source_volatility` is clamped to `[0, 1]` and discounts disagreement only:
/// a frequently changing source is weak evidence that either observer is wrong.
/// Agreement is unchanged. Passing `0` preserves the original model exactly.
pub fn update_pair_trust(
    incoming_trust: f64,
    current_trust: f64,
    agree: bool,
    freshness: f64,
    source_volatility: f64,
) -> PairTrustResult {
    let volatility = clamp01(source_volatility);
    let base = if agree {
        TRUST_AGREE_BASE_GAIN
    } else {
        -TRUST_DISAGREE_BASE_LOSS * (1.0 - volatility)
    };
    PairTrustResult {
        incoming_trust: clamp01(incoming_trust + base * current_trust * freshness),
        current_trust: clamp01(current_trust + base * incoming_trust * freshness),
    }
}

/// Sticky terminal-state reduction plus the public score thresholds.
pub fn next_trust_state(previous: &str, next_score: f64) -> &str {
    if matches!(previous, "suspended" | "revoked") {
        previous
    } else if next_score >= TRUST_TRUSTED_AT {
        "trusted"
    } else if next_score < TRUST_LIMITED_BELOW {
        "limited"
    } else {
        previous
    }
}

pub fn should_reinvestigate(
    freshness: f64,
    churn_count: u64,
    cooldown_until_ms: Option<i64>,
    now_ms: i64,
) -> bool {
    freshness >= VERIFY_CONFLICT_MIN_WEIGHT
        && churn_count < FACT_CHURN_COOLDOWN_THRESHOLD
        && (cooldown_until_ms.is_none()
            || cooldown_until_ms.is_some_and(|deadline| deadline <= now_ms))
}

pub fn reinvestigation_priority(freshness: f64) -> i64 {
    (1_000.0 * freshness).round().max(1.0) as i64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestampSource {
    Carried,
    Fresh,
    Observed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChurnTimestamp {
    pub value_ms: Option<i64>,
    pub source: TimestampSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreviousChurn {
    pub count: u64,
    pub window_started_at_ms: Option<i64>,
    pub has_stored_window: bool,
    pub cooldown_until_ms: Option<i64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChurnState {
    pub count: u64,
    pub window_started_at_ms: i64,
    pub cooldown_until_ms: Option<i64>,
    pub window_started_at: ChurnTimestamp,
    pub cooldown_until: ChurnTimestamp,
}

/// Roll, reset, or cool down a host-supplied churn window.
pub fn next_churn(previous: Option<PreviousChurn>, same: bool, observed_at_ms: i64) -> ChurnState {
    if same {
        let window = match previous {
            Some(previous) if previous.has_stored_window => ChurnTimestamp {
                value_ms: previous.window_started_at_ms,
                source: TimestampSource::Carried,
            },
            _ => ChurnTimestamp {
                value_ms: Some(observed_at_ms),
                source: TimestampSource::Observed,
            },
        };
        let cooldown = previous.and_then(|value| value.cooldown_until_ms);
        return ChurnState {
            count: previous.map_or(0, |value| value.count),
            window_started_at_ms: window.value_ms.unwrap_or(observed_at_ms),
            cooldown_until_ms: cooldown,
            window_started_at: window,
            cooldown_until: ChurnTimestamp {
                value_ms: cooldown,
                source: TimestampSource::Carried,
            },
        };
    }

    let active_window = previous
        .and_then(|value| value.window_started_at_ms)
        .is_some_and(|started| observed_at_ms - started <= FACT_CHURN_WINDOW_MS);
    let count = if active_window {
        previous.map_or(1, |value| value.count + 1)
    } else {
        1
    };
    let started_at_ms = if active_window {
        previous
            .and_then(|value| value.window_started_at_ms)
            .expect("active window has a start")
    } else {
        observed_at_ms
    };
    let fresh_cooldown = count >= FACT_CHURN_COOLDOWN_THRESHOLD;
    let cooldown = if fresh_cooldown {
        Some(observed_at_ms + FACT_COOLDOWN_MS)
    } else {
        previous.and_then(|value| value.cooldown_until_ms)
    };
    ChurnState {
        count,
        window_started_at_ms: started_at_ms,
        cooldown_until_ms: cooldown,
        window_started_at: ChurnTimestamp {
            value_ms: Some(started_at_ms),
            source: TimestampSource::Fresh,
        },
        cooldown_until: ChurnTimestamp {
            value_ms: cooldown,
            source: if fresh_cooldown {
                TimestampSource::Fresh
            } else {
                TimestampSource::Carried
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volatility_discounts_only_disagreement() {
        let stable = update_pair_trust(0.5, 0.5, false, 1.0, 0.0);
        let volatile = update_pair_trust(0.5, 0.5, false, 1.0, 0.75);
        let fully_volatile = update_pair_trust(0.5, 0.5, false, 1.0, 1.0);
        assert_eq!(stable.incoming_trust, 0.48);
        assert_eq!(volatile.incoming_trust, 0.495);
        assert_eq!(fully_volatile.incoming_trust, 0.5);
        assert_eq!(
            update_pair_trust(0.5, 0.5, true, 1.0, 1.0),
            update_pair_trust(0.5, 0.5, true, 1.0, 0.0)
        );
    }

    #[test]
    fn freshness_and_churn_match_the_published_constants() {
        assert_eq!(time_weight(48.0), 0.5);
        let previous = PreviousChurn {
            count: 5,
            window_started_at_ms: Some(1_000),
            has_stored_window: true,
            cooldown_until_ms: None,
        };
        let next = next_churn(Some(previous), false, 2_000);
        assert_eq!(next.count, 6);
        assert_eq!(next.cooldown_until_ms, Some(2_000 + FACT_COOLDOWN_MS));
    }
}
