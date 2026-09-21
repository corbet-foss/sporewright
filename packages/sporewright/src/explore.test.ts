// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { expect, test } from "bun:test";
import { address, AddressedField } from "./field";
import { BatchExplorer, planExploration } from "./explore";

function decision(temperature: number) {
  const field = new AddressedField(["scope", "leaf"]);
  const scope = address({ scope: "system" });
  const leaf = address({ scope: "system", leaf: "x" });
  field.writer("scope")!.setPrior(scope, "known", "cost", 0);
  field.writer("scope")!.setPrior(scope, "unknown", "cost", 1);
  field.writer("leaf")!.observe(leaf, "known", "cost", 0, 0.01);
  const resolved = field.decide(leaf, { temperature });
  if (typeof resolved === "string") throw new Error(resolved);
  return resolved;
}

test("zero temperature executes only the exploitation winner", () => {
  const plan = planExploration(
    decision(0),
    { known: 1, unknown: 1 },
    { max_executions: 3, max_total_cost: 3 },
  );
  if (typeof plan === "string") throw new Error(plan);
  expect(plan.executions).toHaveLength(1);
  expect(plan.executions[0]).toMatchObject({ option: "known", exploratory: false });
});

test("curiosity spends within both count and hard resource cap", () => {
  const curious = decision(2);
  const third = structuredClone(curious.alternatives[1]!);
  third.option = "third";
  third.score_variance = 0.25;
  curious.alternatives.push(third);
  const plan = planExploration(
    curious,
    { known: 1, unknown: 1, third: 1 },
    { max_executions: 3, max_total_cost: 2 },
  );
  if (typeof plan === "string") throw new Error(plan);
  expect(plan.executions).toHaveLength(2);
  expect(plan.total_expected_cost).toBe(2);
  expect(plan.additional_exploration_cost).toBe(1);
  expect(plan.executions[0]!.exploratory).toBeTrue();
});

test("batch curiosity is bounded and rotates across live options", () => {
  const explorer = new BatchExplorer();
  const options = ["browser", "container"];
  expect(explorer.plan(options, 0.35, 50)).toEqual(["browser"]);
  let probes = 1;
  for (let index = 0; index < 99; index += 1) {
    probes += explorer.plan(options, 0.35, 50).length;
  }
  const expected = 100 * 2 * (0.35 / 1.35);
  expect(Math.abs(probes - expected)).toBeLessThanOrEqual(1);
  expect(new BatchExplorer().plan(["container"], 1, 50)).toEqual([]);
  expect(new BatchExplorer().plan(options, 0, 50)).toEqual([]);
});
