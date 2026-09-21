// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { expect, test } from "bun:test";
import { address, AddressedField } from "./field";

test("runtime schemas and snapshots reject ambiguous state", () => {
  expect(() => new AddressedField(["root", "root"])).toThrow("invalid-schema");
  expect(AddressedField.fromJson(JSON.stringify({
    layers: ["root"],
    default_process_variance: [1],
    cells: [{
      address: { root: "x" },
      option: "o",
      dimension: "q",
      evidence: { precision: 0, information: 1, observations: 1 },
    }],
  }))).toBeUndefined();
});

test("the implicit root is a complete field and layer writers cannot write up", () => {
  const field = new AddressedField([]);
  expect(field.rootWriter().setPrior({}, "local", "cost", 2)).toBeUndefined();
  const decision = field.decide({});
  if (typeof decision === "string") throw new Error(decision);
  expect(decision.alternatives[0]!.option).toBe("local");
  expect(decision.alternatives[0]!.dimensions[0]!.contributions).toEqual([{
    address: {},
    posterior_mean: 2,
    posterior_variance: 1,
    residual_mean: 2,
    declared_prior: 2,
    observations: 0,
  }]);

  const layered = new AddressedField(["workspace"]);
  expect(layered.writer("workspace")!.setPrior({}, "local", "cost", 0)).toBe("write-up");
});

test("path residuals add instead of overriding", () => {
  const field = new AddressedField(["b", "c"]);
  const root = address();
  const b = address({ b: "B1" });
  const c = address({ b: "B1", c: "C11" });
  const writer = field.rootWriter();
  writer.setPrior(root, "route", "q1", 1);
  writer.setPrior(b, "route", "q1", 2);
  writer.setPrior(c, "route", "q1", 3);
  const decision = field.decide(c);
  if (typeof decision === "string") throw new Error(decision);
  expect(decision.alternatives[0]!.dimensions[0]!.mean).toBe(6);
  expect(decision.alternatives[0]!.dimensions[0]!.contributions.map(({ declared_prior }) => declared_prior))
    .toEqual([1, 2, 3]);
});

test("feedback updates one path and shared ancestors", () => {
  const field = new AddressedField(["source", "environment"]);
  const root = address();
  const a = address({ source: "a", environment: "browser" });
  const b = address({ source: "b", environment: "browser" });
  field.rootWriter().setPrior(root, "browser", "failure", 0);
  field.writer("environment")!.observe(a, "browser", "failure", 1, 0.1);
  const aDecision = field.decide(a);
  const bDecision = field.decide(b);
  if (typeof aDecision === "string" || typeof bDecision === "string") throw new Error("decision failed");
  const aMean = aDecision.alternatives[0]!.dimensions[0]!.mean;
  const bMean = bDecision.alternatives[0]!.dimensions[0]!.mean;
  expect(aMean).toBeGreaterThan(bMean);
  expect(bMean).toBeGreaterThan(0);
});

test("unrelated population does not change resolution work", () => {
  const field = new AddressedField(["workspace", "task"]);
  field.rootWriter().setPrior(address(), "fast", "cost", 0);
  const target = address({ workspace: "target", task: "one" });
  const before = field.decide(target);
  if (typeof before === "string") throw new Error(before);
  for (let index = 0; index < 10_000; index++) {
    field.writer("task")!.setPrior(
      address({ workspace: `w${index}`, task: "one" }),
      "slow",
      "cost",
      1,
    );
  }
  const after = field.decide(target);
  if (typeof after === "string") throw new Error(after);
  expect(after.work).toEqual(before.work);
});

test("decideAmong bounds both candidates and work to the host allow-list", () => {
  const field = new AddressedField(["workspace", "task"]);
  const target = address({ workspace: "target", task: "one" });
  field.rootWriter().setPrior(address(), "allowed", "cost", 0);
  const before = field.decideAmong(target, ["allowed"]);
  if (typeof before === "string") throw new Error(before);
  for (let index = 0; index < 1_000; index++) {
    field.rootWriter().setPrior(address(), `historical-${index}`, "cost", index + 1);
  }
  const after = field.decideAmong(target, ["allowed", "absent"]);
  if (typeof after === "string") throw new Error(after);
  expect(after.alternatives.map(({ option }) => option)).toEqual(["allowed"]);
  expect(after.work).toEqual({
    ...before.work,
    // The absent allow-listed option costs a bounded lookup at the one root
    // prefix; the thousand unrelated stored options are never visited.
    parameter_lookups: before.work.parameter_lookups + 1,
  });
});

test("hard veto cannot be cancelled by curiosity", () => {
  const field = new AddressedField(["environment"]);
  const root = address();
  const local = address({ environment: "browser" });
  const writer = field.rootWriter();
  writer.setPrior(root, "browser", "cost", 0);
  writer.setPrior(root, "container", "cost", 1);
  writer.setGate(local, "browser", "reachable", false);
  expect(field.resolve(local, { temperature: 1_000 })).toEqual(["container"]);
});

test("snapshot rebuilds derived messages", () => {
  const field = new AddressedField(["leaf"]);
  const root = address();
  const leaf = address({ leaf: "a" });
  field.rootWriter().setPrior(root, "route", "cost", 0);
  field.writer("leaf")!.observe(leaf, "route", "cost", 2, 0.5);
  const rebuilt = AddressedField.fromJson(field.toJson())!;
  expect(rebuilt.toJson()).toBe(field.toJson());
  expect(rebuilt.decide(leaf)).toEqual(field.decide(leaf));
});

test("retired subtree compacts exactly for ancestors and siblings", () => {
  const field = new AddressedField(["source", "task", "environment"]);
  const root = address();
  const retired = address({ source: "portal", task: "old" });
  const observed = address({
    source: "portal",
    task: "old",
    environment: "browser",
  });
  const sibling = address({
    source: "portal",
    task: "new",
    environment: "browser",
  });
  const writer = field.rootWriter();
  writer.setPrior(root, "browser", "failure", 0);
  writer.observe(observed, "browser", "failure", 8, 0.2);
  const before = field.decide(sibling);
  if (typeof before === "string") throw new Error(before);
  const beforeDimension = before.alternatives[0]!.dimensions[0]!;
  const cellsBefore = field.storedCellCount();
  const trace = writer.compactSubtree(retired, "browser", "failure");
  if (typeof trace === "string") throw new Error(trace);
  const after = field.decide(sibling);
  if (typeof after === "string") throw new Error(after);
  const afterDimension = after.alternatives[0]!.dimensions[0]!;

  expect(Math.abs(beforeDimension.mean - afterDimension.mean)).toBeLessThan(1e-12);
  expect(Math.abs(beforeDimension.variance - afterDimension.variance)).toBeLessThan(1e-12);
  expect(trace.observations).toBe(1);
  expect(trace.removed_cells).toBe(2);
  expect(field.storedCellCount()).toBeLessThan(cellsBefore);
  expect(field.setDefaultProcessVariance("task", 2)).toBe("frozen-process-variance");
  expect(AddressedField.fromJson(field.toJson())!.toJson()).toBe(field.toJson());
});

test("discount reopens uncertainty without a clock in the core", () => {
  const field = new AddressedField(["leaf"]);
  const root = address();
  const leaf = address({ leaf: "changing" });
  const writer = field.rootWriter();
  writer.setPrior(root, "route", "cost", 0);
  writer.observe(leaf, "route", "cost", 4, 0.01);
  const before = field.decide(leaf);
  if (typeof before === "string") throw new Error(before);
  const trace = writer.discountEvidence(leaf, "route", "cost", 0);
  if (typeof trace === "string") throw new Error(trace);
  const after = field.decide(leaf);
  if (typeof after === "string") throw new Error(after);
  expect(after.alternatives[0]!.dimensions[0]!.variance)
    .toBeGreaterThan(before.alternatives[0]!.dimensions[0]!.variance);
  expect(Math.abs(after.alternatives[0]!.dimensions[0]!.mean))
    .toBeLessThan(Math.abs(before.alternatives[0]!.dimensions[0]!.mean));
  expect(trace.precision_after).toBe(0);
  expect(writer.discountEvidence(leaf, "route", "cost", 1.1)).toBe("invalid-discount");
});
