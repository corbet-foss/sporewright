// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// Unit tests for the declarative tensor config. Mirrors the Rust `config.rs` tests:
// the declarative builder must produce a tensor BYTE-IDENTICAL to the imperative
// seeding (the faithfulness claim), and write-down / unknown-level errors must surface
// exactly as the imperative `Writer` calls would.

import { describe, expect, it } from "bun:test";
import {
  bytesCell,
  instantiate,
  optionWeightCell,
  tensorConfig,
  Tensor,
  valueCell,
  weightCell,
  withCell,
  type TensorConfig,
  type WriteError,
} from "./index";

const GLOBAL = "global";
const SOURCE = "source";
const JOB = "job";
const FINANCIAL = "financial";
const LATENCY = "latency";
const RELIABILITY = "reliability";

type FleetEntry = [id: string, financial: number];

/** Imperatively seed a fleet exactly as `router.rs::seed_fleet` does. */
function seedFleetImperative(t: Tensor, fleet: FleetEntry[]): void {
  const w = t.writer(GLOBAL)!;
  w.setWeight(GLOBAL, "", FINANCIAL, 1.0);
  w.setWeight(GLOBAL, "", LATENCY, 0.001);
  w.setWeight(GLOBAL, "", RELIABILITY, 4.0);
  for (const [id, financial] of fleet) w.setValue(GLOBAL, "", id, FINANCIAL, financial);
}

/** The SAME seeding as a `TensorConfig`, cell for cell, in the SAME order. */
function seedFleetConfig(fleet: FleetEntry[]): TensorConfig {
  let cfg = tensorConfig([GLOBAL, SOURCE, JOB], [
    weightCell(GLOBAL, GLOBAL, "", FINANCIAL, 1.0),
    weightCell(GLOBAL, GLOBAL, "", LATENCY, 0.001),
    weightCell(GLOBAL, GLOBAL, "", RELIABILITY, 4.0),
  ]);
  for (const [id, financial] of fleet) {
    cfg = withCell(cfg, valueCell(GLOBAL, GLOBAL, "", id, FINANCIAL, financial));
  }
  return cfg;
}

function fleet(): FleetEntry[] {
  return [
    ["dev:browser:b1", 0.0],
    ["dev:container:c1", 0.1],
    ["dev:render:r1", 1.0],
  ];
}

describe("declarative tensor config — instantiate(config) is the imperative seeding, as data", () => {
  it("instantiation is byte-identical to imperative seeding (faithfulness)", () => {
    const f = fleet();
    const imperative = new Tensor([GLOBAL, SOURCE, JOB]);
    seedFleetImperative(imperative, f);

    const declared = instantiate(seedFleetConfig(f));
    expect(declared).toBeInstanceOf(Tensor);
    // The wire form is the byte witness — identical bytes ⇒ identical tensor.
    expect((declared as Tensor).toJson()).toEqual(imperative.toJson());
  });

  it("declared and imperative resolve identically (decision-equivalence)", () => {
    const f = fleet();
    const imperative = new Tensor([GLOBAL, SOURCE, JOB]);
    seedFleetImperative(imperative, f);
    const declared = instantiate(seedFleetConfig(f)) as Tensor;

    expect(declared.resolve({})).toEqual(imperative.resolve({}));
    expect(declared.resolve({})).toEqual([
      "dev:browser:b1",
      "dev:container:c1",
      "dev:render:r1",
    ]);
  });

  it("all four seed kinds round-trip through a config", () => {
    const doc = new Uint8Array([0, 1, 2, 255, 60, 62]);
    const cfg = tensorConfig([GLOBAL, "workspace"], [
      valueCell(GLOBAL, GLOBAL, "", "groq", "latency", 0.4),
      weightCell(GLOBAL, GLOBAL, "", "latency", 1.0),
      optionWeightCell(GLOBAL, GLOBAL, "", "groq", "latency", 2.0),
      bytesCell("workspace", "workspace", "acme", "doc:cv", "state", doc),
    ]);

    const imp = new Tensor([GLOBAL, "workspace"]);
    const wg = imp.writer(GLOBAL)!;
    wg.setValue(GLOBAL, "", "groq", "latency", 0.4);
    wg.setWeight(GLOBAL, "", "latency", 1.0);
    wg.setOptionWeight(GLOBAL, "", "groq", "latency", 2.0);
    imp.writer("workspace")!.setBytes("workspace", "acme", "doc:cv", "state", doc);

    const built = instantiate(cfg) as Tensor;
    expect(built.toJson()).toEqual(imp.toJson());

    expect(built.bytes({ workspace: "acme" }, "doc:cv", "state")).toEqual(doc);
    expect(built.value({}, "groq", "latency")).toBe(0.4);
    expect(built.weight({}, "groq", "latency")).toBe(2.0); // per-option
  });

  it("a gate seeds through the value path (no special kind)", () => {
    const cfg = tensorConfig([GLOBAL, JOB], [
      weightCell(GLOBAL, GLOBAL, "", "latency", 1.0),
      valueCell(GLOBAL, GLOBAL, "", "keep", "latency", 0.2),
      valueCell(GLOBAL, GLOBAL, "", "drop", "latency", 0.1),
      valueCell(JOB, JOB, "j1", "drop", "priv:residential", Infinity),
    ]);
    const t = instantiate(cfg) as Tensor;
    expect(t.resolve({ job: "j1" })).toEqual(["keep"]);
  });

  it("a write-up cell is rejected like the imperative call", () => {
    const cfg = tensorConfig([GLOBAL, JOB], [
      valueCell(JOB, GLOBAL, "", "x", "latency", 0.1),
    ]);
    expect(instantiate(cfg)).toBe("write-up" satisfies WriteError);
  });

  it("an unknown level is rejected", () => {
    const cfg = tensorConfig([GLOBAL], [valueCell("nope", "nope", "", "x", "latency", 0.1)]);
    expect(instantiate(cfg)).toBe("unknown-level" satisfies WriteError);
    const cfg2 = tensorConfig([GLOBAL], [valueCell(GLOBAL, "missing", "", "x", "latency", 0.1)]);
    expect(instantiate(cfg2)).toBe("unknown-level" satisfies WriteError);
  });

  it("a kind/value mismatch is rejected as bad-value-kind (not unknown-level)", () => {
    // A `bytes` cell carrying an `{ f64 }` payload is a malformed CELL, not an unknown
    // TIER (all tiers are valid) → "bad-value-kind". Cross-core mirror of the Rust test.
    const cfg = tensorConfig([GLOBAL], [
      { floor: GLOBAL, level: GLOBAL, inst: "", option: "x", dim: "doc", kind: "bytes", value: { f64: 1.0 } },
    ]);
    expect(instantiate(cfg)).toBe("bad-value-kind" satisfies WriteError);
  });
});
