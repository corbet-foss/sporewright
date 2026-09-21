// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Budgeted curiosity plans over an addressed-field decision.

use crate::field::FieldDecision;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationBudget {
    /// Maximum routes executed for one logical request.
    pub max_executions: usize,
    /// Hard total resource ceiling in the product's declared cost unit.
    pub max_total_cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedExecution {
    pub option: String,
    pub expected_cost: f64,
    pub exploratory: bool,
    pub uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub executions: Vec<PlannedExecution>,
    pub exploitation_baseline: Option<String>,
    pub baseline_cost: f64,
    pub total_expected_cost: f64,
    pub additional_exploration_cost: f64,
    pub budget: ExplorationBudget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlanError {
    InvalidBudget,
    InvalidCost,
}

/// A constant-memory exploration scheduler for repeated batches.
///
/// The token bucket converts a non-negative temperature into the bounded rate
/// `temperature / (1 + temperature)`.  Credits accrue per live option, while a
/// round-robin cursor prevents an option from being permanently hidden by
/// deterministic tie-breaking.  The first batch with at least two options gets
/// one bootstrap probe; after that the long-run number of probes is bounded by
/// the configured rate.  Products remain responsible for present-tense vetoes
/// and may defer a planned probe until that option is viable for a task.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchExplorer {
    cursor: usize,
    credit: f64,
    bootstrapped: bool,
}

impl BatchExplorer {
    /// Reserve exploration options for one batch.
    ///
    /// Input order is irrelevant: options are sorted and deduplicated before a
    /// rotating slice is selected. Memory is `O(1)` and planning is `O(K log K)`
    /// for `K` live options, independent of actors, tasks, or historical events.
    pub fn plan<I, S>(&mut self, options: I, temperature: f64, batch_slots: usize) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if !temperature.is_finite() || temperature <= 0.0 || batch_slots == 0 {
            return Vec::new();
        }
        let mut options: Vec<String> = options.into_iter().map(Into::into).collect();
        options.sort();
        options.dedup();
        if options.len() < 2 {
            return Vec::new();
        }

        let rate = temperature / (1.0 + temperature);
        self.credit = (self.credit + rate * options.len() as f64).min(options.len() as f64);
        if !self.bootstrapped {
            self.credit = self.credit.max(1.0);
        }
        let count = (self.credit.floor() as usize)
            .min(batch_slots)
            .min(options.len());
        if count == 0 {
            return Vec::new();
        }

        let mut planned = Vec::with_capacity(count);
        for offset in 0..count {
            planned.push(options[(self.cursor + offset) % options.len()].clone());
        }
        self.cursor = (self.cursor + count) % options.len();
        self.credit -= count as f64;
        self.bootstrapped = true;
        planned
    }
}

fn fcmp(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

/// Build a bounded multi-execution plan.
///
/// `expected_costs` uses a product-declared unit (tokens, money, joules, or
/// another non-negative resource). At temperature zero only the exploitation
/// winner is admitted, regardless of a larger count budget. At positive
/// temperature, additional probes are ranked by uncertainty gained per unit of
/// cost after the curiosity-selected primary route.
pub fn plan_exploration(
    decision: &FieldDecision,
    expected_costs: &BTreeMap<String, f64>,
    budget: ExplorationBudget,
) -> Result<ExecutionPlan, PlanError> {
    if budget.max_executions == 0
        || !budget.max_total_cost.is_finite()
        || budget.max_total_cost < 0.0
    {
        return Err(PlanError::InvalidBudget);
    }
    if expected_costs
        .values()
        .any(|cost| !cost.is_finite() || *cost < 0.0)
    {
        return Err(PlanError::InvalidCost);
    }

    let viable: Vec<_> = decision
        .alternatives
        .iter()
        .filter(|candidate| candidate.viable)
        .filter_map(|candidate| {
            expected_costs
                .get(&candidate.option)
                .copied()
                .map(|cost| (candidate, cost))
        })
        .collect();
    let exploitation = viable.iter().min_by(|(left, _), (right, _)| {
        fcmp(left.expected_score, right.expected_score).then_with(|| left.option.cmp(&right.option))
    });
    let exploitation_baseline = exploitation.map(|(candidate, _)| candidate.option.clone());
    let baseline_cost = exploitation.map(|(_, cost)| *cost).unwrap_or(0.0);

    let limit = if decision.policy.temperature == 0.0 {
        1
    } else {
        budget.max_executions
    };
    let mut executions = Vec::new();
    let mut total = 0.0;
    if let Some((primary, cost)) = viable.first() {
        if *cost <= budget.max_total_cost {
            executions.push(PlannedExecution {
                option: primary.option.clone(),
                expected_cost: *cost,
                exploratory: exploitation_baseline.as_deref() != Some(primary.option.as_str()),
                uncertainty: primary.score_variance.sqrt(),
            });
            total = *cost;
        }
    }

    if limit > executions.len() && decision.policy.temperature > 0.0 {
        let selected: std::collections::BTreeSet<&str> = executions
            .iter()
            .map(|execution| execution.option.as_str())
            .collect();
        let mut probes: Vec<_> = viable
            .iter()
            .filter(|(candidate, _)| !selected.contains(candidate.option.as_str()))
            .copied()
            .collect();
        probes.sort_by(|(left, left_cost), (right, right_cost)| {
            let left_density = left.score_variance.sqrt() / left_cost.max(f64::EPSILON);
            let right_density = right.score_variance.sqrt() / right_cost.max(f64::EPSILON);
            fcmp(right_density, left_density).then_with(|| left.option.cmp(&right.option))
        });
        for (candidate, cost) in probes {
            if executions.len() >= limit {
                break;
            }
            if total + cost > budget.max_total_cost {
                continue;
            }
            executions.push(PlannedExecution {
                option: candidate.option.clone(),
                expected_cost: cost,
                exploratory: true,
                uncertainty: candidate.score_variance.sqrt(),
            });
            total += cost;
        }
    }

    Ok(ExecutionPlan {
        executions,
        exploitation_baseline,
        baseline_cost,
        total_expected_cost: total,
        additional_exploration_cost: (total - baseline_cost).max(0.0),
        budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{address, AddressedField, DecisionPolicy};

    fn uncertain_decision(temperature: f64) -> FieldDecision {
        let mut field = AddressedField::new(["scope", "leaf"]);
        let scope = address([("scope", "system")]);
        let leaf = address([("scope", "system"), ("leaf", "x")]);
        let mut writer = field.writer("scope").unwrap();
        writer.set_prior(&scope, "known", "cost", 0.0).unwrap();
        writer.set_prior(&scope, "unknown", "cost", 1.0).unwrap();
        writer.observe(&leaf, "known", "cost", 0.0, 0.01).unwrap();
        field.decide(&leaf, DecisionPolicy { temperature }).unwrap()
    }

    #[test]
    fn zero_temperature_executes_only_the_exploitation_winner() {
        let decision = uncertain_decision(0.0);
        let costs = BTreeMap::from([("known".to_owned(), 1.0), ("unknown".to_owned(), 1.0)]);
        let plan = plan_exploration(
            &decision,
            &costs,
            ExplorationBudget {
                max_executions: 3,
                max_total_cost: 3.0,
            },
        )
        .unwrap();
        assert_eq!(plan.executions.len(), 1);
        assert_eq!(plan.executions[0].option, "known");
        assert!(!plan.executions[0].exploratory);
    }

    #[test]
    fn curiosity_can_spend_three_calls_but_never_crosses_the_hard_cap() {
        let mut decision = uncertain_decision(2.0);
        let mut third = decision.alternatives[1].clone();
        third.option = "third".to_owned();
        third.score_variance = 0.25;
        decision.alternatives.push(third);
        let costs = BTreeMap::from([
            ("known".to_owned(), 1.0),
            ("unknown".to_owned(), 1.0),
            ("third".to_owned(), 1.0),
        ]);
        let plan = plan_exploration(
            &decision,
            &costs,
            ExplorationBudget {
                max_executions: 3,
                max_total_cost: 2.0,
            },
        )
        .unwrap();
        assert_eq!(plan.executions.len(), 2);
        assert_eq!(plan.total_expected_cost, 2.0);
        assert_eq!(plan.additional_exploration_cost, 1.0);
        assert!(plan.executions[0].exploratory);
    }

    #[test]
    fn batch_curiosity_bootstraps_then_tracks_a_bounded_rate() {
        let mut explorer = BatchExplorer::default();
        let options = ["browser", "container"];
        assert_eq!(explorer.plan(options, 0.35, 50), ["browser"]);

        let mut probes = 1;
        for _ in 0..99 {
            probes += explorer.plan(options, 0.35, 50).len();
        }
        let expected = 100.0 * 2.0 * (0.35 / 1.35);
        assert!((probes as f64 - expected).abs() <= 1.0);
    }

    #[test]
    fn batch_curiosity_rotates_and_never_spends_without_alternatives() {
        let mut explorer = BatchExplorer::default();
        assert!(explorer.plan(["container"], 1.0, 50).is_empty());
        assert_eq!(
            explorer.plan(["desktop", "browser", "container"], 0.01, 50),
            ["browser"]
        );
        for _ in 0..33 {
            let _ = explorer.plan(["desktop", "browser", "container"], 0.01, 50);
        }
        assert_eq!(
            explorer.plan(["desktop", "browser", "container"], 0.01, 50),
            ["container"]
        );
        assert!(explorer.plan(["browser", "container"], 0.0, 50).is_empty());
    }
}
