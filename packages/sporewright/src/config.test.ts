// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { describe, expect, it } from "bun:test";
import {
  bytesCell,
  instantiate,
  optionWeightCell,
  scope,
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

function declaredFleet(): TensorConfig {
  let config = tensorConfig([GLOBAL, SOURCE, JOB], [
    weightCell(GLOBAL, GLOBAL, {}, "financial", 1),
    weightCell(GLOBAL, GLOBAL, {}, "latency", 0.001),
  ]);
  config = withCell(config, valueCell(GLOBAL, GLOBAL, {}, "browser", "financial", 0));
  config = withCell(config, valueCell(GLOBAL, GLOBAL, {}, "native", "financial", 0.1));
  return config;
}

describe("declarative scoped tensor config", () => {
  it("is byte-identical to imperative seeding", () => {
    const imperative = new Tensor([GLOBAL, SOURCE, JOB]);
    const writer = imperative.writer(GLOBAL)!;
    writer.setWeight(GLOBAL, {}, "financial", 1);
    writer.setWeight(GLOBAL, {}, "latency", 0.001);
    writer.setValue(GLOBAL, {}, "browser", "financial", 0);
    writer.setValue(GLOBAL, {}, "native", "financial", 0.1);

    const declared = instantiate(declaredFleet());
    expect(declared).toBeInstanceOf(Tensor);
    expect((declared as Tensor).toJson()).toBe(imperative.toJson());
  });

  it("preserves every seed kind and nested scope", () => {
    const document = new Uint8Array([0, 1, 2, 255]);
    const config = tensorConfig([GLOBAL, SOURCE], [
      valueCell(GLOBAL, GLOBAL, {}, "browser", "latency", 0.4),
      weightCell(GLOBAL, GLOBAL, {}, "latency", 1),
      optionWeightCell(GLOBAL, GLOBAL, {}, "browser", "latency", 2),
      bytesCell(SOURCE, SOURCE, scope({ source: "nzz" }), "document", "state", document),
    ]);
    const built = instantiate(config) as Tensor;
    expect(built.bytes(scope({ source: "nzz" }), "document", "state")).toEqual(document);
    expect(built.weight({}, "browser", "latency")).toBe(2);
  });

  it("rejects write-up, unknown layers, deep scope and kind mismatch", () => {
    expect(instantiate(tensorConfig([GLOBAL, JOB], [
      valueCell(JOB, GLOBAL, {}, "x", "latency", 0.1),
    ]))).toBe("write-up" satisfies WriteError);

    expect(instantiate(tensorConfig([GLOBAL], [
      valueCell("missing", GLOBAL, {}, "x", "latency", 0.1),
    ]))).toBe("unknown-layer" satisfies WriteError);

    expect(instantiate(tensorConfig([GLOBAL, JOB], [
      valueCell(GLOBAL, GLOBAL, scope({ job: "j1" }), "x", "latency", 0.1),
    ]))).toBe("scope-outside-layer" satisfies WriteError);

    expect(instantiate({
      layers: [GLOBAL],
      seeds: [{
        floor: GLOBAL,
        layer: GLOBAL,
        scope: {},
        option: "x",
        dim: "doc",
        kind: "bytes",
        value: { f64: 1 },
      }],
    })).toBe("bad-value-kind" satisfies WriteError);
  });
});
