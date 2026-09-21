// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/** Budgeted curiosity plans over an addressed-field decision. */

import type { FieldDecision } from "./field";

export interface ExplorationBudget {
  max_executions: number;
  max_total_cost: number;
}

export interface PlannedExecution {
  option: string;
  expected_cost: number;
  exploratory: boolean;
  uncertainty: number;
}

export interface ExecutionPlan {
  executions: PlannedExecution[];
  exploitation_baseline?: string;
  baseline_cost: number;
  total_expected_cost: number;
  additional_exploration_cost: number;
  budget: ExplorationBudget;
}

export type PlanError = "invalid-budget" | "invalid-cost";

/**
 * Constant-memory exploration scheduler for repeated batches.
 *
 * Credits accrue at `temperature / (1 + temperature)` per live option. A
 * round-robin cursor prevents deterministic tie-breaking from starving an
 * option, and the first batch with alternatives receives one bootstrap probe.
 */
export class BatchExplorer {
  private cursor = 0;
  private credit = 0;
  private bootstrapped = false;

  plan(optionsInput: Iterable<string>, temperature: number, batchSlots: number): string[] {
    if (!Number.isFinite(temperature)
      || temperature <= 0
      || !Number.isSafeInteger(batchSlots)
      || batchSlots <= 0) return [];
    const options = [...new Set(optionsInput)].sort(compareStrings);
    if (options.length < 2) return [];

    const rate = temperature / (1 + temperature);
    this.credit = Math.min(options.length, this.credit + rate * options.length);
    if (!this.bootstrapped) this.credit = Math.max(1, this.credit);
    const count = Math.min(Math.floor(this.credit), batchSlots, options.length);
    if (count === 0) return [];

    const planned = Array.from(
      { length: count },
      (_, offset) => options[(this.cursor + offset) % options.length]!,
    );
    this.cursor = (this.cursor + count) % options.length;
    this.credit -= count;
    this.bootstrapped = true;
    return planned;
  }
}

function compareStrings(left: string, right: string): number {
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const a = left.codePointAt(i)!;
    const b = right.codePointAt(j)!;
    if (a !== b) return a < b ? -1 : 1;
    i += a > 0xffff ? 2 : 1;
    j += b > 0xffff ? 2 : 1;
  }
  return i < left.length ? 1 : j < right.length ? -1 : 0;
}

export function planExploration(
  decision: FieldDecision,
  expectedCosts: Record<string, number>,
  budget: ExplorationBudget,
): ExecutionPlan | PlanError {
  if (!Number.isSafeInteger(budget.max_executions)
    || budget.max_executions <= 0
    || !Number.isFinite(budget.max_total_cost)
    || budget.max_total_cost < 0) return "invalid-budget";
  if (Object.values(expectedCosts).some((cost) => !Number.isFinite(cost) || cost < 0)) return "invalid-cost";

  const viable = decision.alternatives
    .filter(({ viable, option }) => viable && expectedCosts[option] !== undefined)
    .map((candidate) => [candidate, expectedCosts[candidate.option]!] as const);
  const exploitation = [...viable].sort(([left], [right]) =>
    left.expected_score - right.expected_score || compareStrings(left.option, right.option)
  )[0];
  const exploitationBaseline = exploitation?.[0].option;
  const baselineCost = exploitation?.[1] ?? 0;
  const limit = decision.policy.temperature === 0 ? 1 : budget.max_executions;
  const executions: PlannedExecution[] = [];
  let total = 0;

  const primary = viable[0];
  if (primary !== undefined && primary[1] <= budget.max_total_cost) {
    executions.push({
      option: primary[0].option,
      expected_cost: primary[1],
      exploratory: primary[0].option !== exploitationBaseline,
      uncertainty: Math.sqrt(primary[0].score_variance),
    });
    total = primary[1];
  }

  if (executions.length < limit && decision.policy.temperature > 0) {
    const selected = new Set(executions.map(({ option }) => option));
    const probes = viable
      .filter(([candidate]) => !selected.has(candidate.option))
      .sort(([left, leftCost], [right, rightCost]) =>
        Math.sqrt(right.score_variance) / Math.max(rightCost, Number.EPSILON)
        - Math.sqrt(left.score_variance) / Math.max(leftCost, Number.EPSILON)
        || compareStrings(left.option, right.option)
      );
    for (const [candidate, cost] of probes) {
      if (executions.length >= limit) break;
      if (total + cost > budget.max_total_cost) continue;
      executions.push({
        option: candidate.option,
        expected_cost: cost,
        exploratory: true,
        uncertainty: Math.sqrt(candidate.score_variance),
      });
      total += cost;
    }
  }

  return {
    executions,
    exploitation_baseline: exploitationBaseline,
    baseline_cost: baselineCost,
    total_expected_cost: total,
    additional_exploration_cost: Math.max(0, total - baselineCost),
    budget: { ...budget },
  };
}
