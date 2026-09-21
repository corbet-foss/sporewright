// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { expect, test } from "bun:test";
import { Tensor } from "./index";

test("fold/resolve and write-down", () => {
  const t = new Tensor(["global", "workspace", "device"]);
  const o = t.writer("global")!;
  o.setValue("global", "", "groq", "latency", 0.4);
  o.setValue("global", "", "cerebras", "latency", 0.3);
  o.setWeight("global", "", "latency", 1.0);
  expect(t.resolve({})).toEqual(["cerebras", "groq"]);

  t.writer("device")!.setValue("device", "A", "groq", "latency", 0.05);
  expect(t.resolve({ device: "A" })).toEqual(["groq", "cerebras"]);
  expect(t.writer("device")!.setValue("global", "", "groq", "latency", 0.0)).toBe("write-up");
  expect(t.writer("nope")).toBeUndefined();
});

test("per-option weight is the gas pedal", () => {
  const t = new Tensor(["global"]);
  const o = t.writer("global")!;
  o.setValue("global", "", "groq", "latency", 0.2);
  o.setValue("global", "", "cerebras", "latency", 0.3);
  o.setWeight("global", "", "latency", 1.0);
  expect(t.resolve({})).toEqual(["groq", "cerebras"]);
  t.writer("global")!.setOptionWeight("global", "", "groq", "latency", 10.0);
  expect(t.resolve({})).toEqual(["cerebras", "groq"]);
});

test("corroborate needs independent reporters", () => {
  const t = new Tensor(["global", "device"]);
  for (const r of ["A", "B", "C"]) t.writer("device")!.setValue("device", r, "D", "can_scrape", 1.0);
  t.writer("device")!.setValue("device", "E", "E", "can_magic", 1.0); // faker self-claim
  t.writer("device")!.setValue("device", "D", "D", "can_scrape", 1.0); // self-vote, excluded
  t.corroborate("device", "global", "", 2);
  expect(t.value({}, "D", "can_scrape")).toBe(1.0);
  expect(t.value({}, "E", "can_magic")).toBeUndefined();
});

test("reduce median rolls value up (byte-equivalent)", () => {
  const t = new Tensor(["global", "device"]);
  t.writer("device")!.setValue("device", "A", "groq", "latency", 0.2);
  t.writer("device")!.setValue("device", "B", "groq", "latency", 0.4);
  t.rollUp("device", "global", "");
  expect(t.value({}, "groq", "latency")).toBe(0.30000000000000004);
});

test("a document rides the tensor, inert to routing, byte-identical on the wire", () => {
  const a = new Tensor(["global", "workspace"]);
  const w = a.writer("workspace")!;
  const doc = new Uint8Array([0, 1, 2, 255, 60, 62]);
  w.setBytes("workspace", "acme", "doc:cv", "state", doc);
  w.setValue("workspace", "acme", "groq", "latency", 0.2);
  w.setWeight("workspace", "acme", "latency", 1.0);
  const at = { workspace: "acme" };
  expect(a.bytes(at, "doc:cv", "state")).toEqual(doc);
  expect(a.value(at, "doc:cv", "state")).toBeUndefined();
  expect(a.resolve(at)).toEqual(["groq"]);

  const json = a.toJson();
  expect(json).toContain('"b64":"AAEC/zw+"'); // same base64 as the Rust core
  const b = Tensor.fromJson(json)!;
  expect(b.bytes(at, "doc:cv", "state")).toEqual(doc);
  expect(b.value(at, "groq", "latency")).toBe(0.2);
});

test("json round-trips value, weight, gate", () => {
  const a = new Tensor(["global", "device"]);
  const o = a.writer("global")!;
  o.setValue("global", "", "groq", "latency", 0.4);
  o.setWeight("global", "", "latency", 1.0);
  o.setOptionWeight("global", "", "groq", "latency", 2.0);
  a.writer("device")!.setValue("device", "A", "peer:X", "reach", Infinity);

  const json = a.toJson();
  const b = Tensor.fromJson(json)!;
  expect(b.value({}, "groq", "latency")).toBe(0.4);
  expect(b.weight({}, "cerebras", "latency")).toBe(1.0); // shared
  expect(b.weight({}, "groq", "latency")).toBe(2.0); // per-option
  expect(b.value({ device: "A" }, "peer:X", "reach")).toBe(Infinity); // gate survived
  expect(b.toJson()).toBe(json); // canonical
});

test("applyJson tolerates a malformed cell", () => {
  const cells =
    '[{"level":"global","inst":"","option":"groq","dim":"latency","v":0.1},' +
    '{"level":"global","inst":"","option":"bad","dim":"latency","v":true},' +
    '{"level":"global","inst":"","option":"cerebras","dim":"latency","v":0.2}]';
  const t = new Tensor(["global"]);
  expect(() => t.applyJson(cells)).not.toThrow();
  expect(t.value({}, "groq", "latency")).toBe(0.1);
  expect(t.value({}, "cerebras", "latency")).toBe(0.2);
  expect(t.value({}, "bad", "latency")).toBeUndefined();
});

test("applyJson drops a cell with a malformed weight layer WHOLE (Rust-parity)", () => {
  // PARITY: a valid `v` but a malformed `w` must drop the ENTIRE cell — v must NOT
  // land. Rust decodes the JsonCell atomically; the TS applyJson now decodes both
  // layers before putting either. Both cores yield value(x, lat) === undefined.
  const cells = '[{"level":"global","inst":"","option":"x","dim":"lat","v":0.5,"w":true}]';
  const t = new Tensor(["global"]);
  expect(() => t.applyJson(cells)).not.toThrow();
  expect(t.value({}, "x", "lat")).toBeUndefined();
  expect(t.weight({}, "x", "lat")).toBeUndefined();
});

test("clearLevelInst removes one slice and spares the others", () => {
  const t = new Tensor(["global", "job"]);
  const w = t.writer("job")!;
  w.setValue("job", "i", "opt", "priv:cap", Infinity);
  w.setValue("job", "i", "opt", "lat", 0.2);
  w.setValue("job", "j", "opt", "lat", 0.3);
  expect(t.value({ job: "i" }, "opt", "lat")).toBe(0.2);

  t.clearLevelInst("job", "i");
  expect(t.value({ job: "i" }, "opt", "lat")).toBeUndefined();
  expect(t.value({ job: "i" }, "opt", "priv:cap")).toBeUndefined();
  // (job, j) survives.
  expect(t.value({ job: "j" }, "opt", "lat")).toBe(0.3);
});

test("reduceSum adds value up (the budget feeder, byte-equivalent)", () => {
  const t = new Tensor(["global", "device"]);
  t.writer("device")!.setValue("device", "A", "groq", "tokens", 200.0);
  t.writer("device")!.setValue("device", "B", "groq", "tokens", 350.0);
  t.reduceSum("device", "global", "", "groq", "tokens");
  // Total draw on a shared pool is the SUM (the budget feeder), not the median.
  expect(t.value({}, "groq", "tokens")).toBe(550.0);
});

test("a non-finite value gates the option (+inf, -inf, NaN alike)", () => {
  const t = new Tensor(["global"]);
  const o = t.writer("global")!;
  o.setWeight("global", "", "latency", 1.0);
  o.setValue("global", "", "keep", "latency", 0.2);
  o.setValue("global", "", "pos", "latency", 0.1);
  o.setValue("global", "", "neg", "latency", 0.05);
  o.setValue("global", "", "nanopt", "latency", 0.04);
  o.setValue("global", "", "pos", "reach", Infinity);
  o.setValue("global", "", "neg", "reach", -Infinity);
  o.setValue("global", "", "nanopt", "reach", NaN);
  // Only the finite option survives: -inf does not anti-gate, NaN does not linger.
  expect(t.resolve({})).toEqual(["keep"]);
});
