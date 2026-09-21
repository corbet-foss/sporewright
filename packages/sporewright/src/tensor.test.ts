// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { expect, test } from "bun:test";
import { scope, Tensor } from "./index";

test("scoped layers fold most-specific preferences", () => {
  const tensor = new Tensor(["session", "source", "job", "device"]);
  const root = tensor.writer("session")!;
  root.setWeight("session", {}, "latency", 1);
  root.setValue("session", {}, "browser", "latency", 0.1);
  root.setValue("session", {}, "native", "latency", 0.2);

  tensor.writer("device")!.setValue(
    "device",
    scope({ source: "nzz", device: "A" }),
    "browser",
    "latency",
    0.8,
  );

  expect(tensor.resolve(scope({ source: "nzz", device: "A" }))).toEqual(["native", "browser"]);
  expect(tensor.resolve(scope({ source: "proton", device: "A" }))).toEqual(["browser", "native"]);
  expect(tensor.writer("device")!.setValue("session", {}, "browser", "latency", 0)).toBe("write-up");
});

test("feedback changes only its owning preference layer and scope", () => {
  const tensor = new Tensor(["workspace", "stage", "consumer", "instance", "device"]);
  tensor.writer("workspace")!.setValue("workspace", {}, "groq", "quality", 0.2);
  const writer = tensor.writer("consumer")!;
  const fit = scope({ stage: "evaluate", consumer: "fit" });
  const change = writer.nudgeValue("consumer", fit, "groq", "quality", 1, 0.5);

  expect(typeof change).not.toBe("string");
  expect(tensor.value(fit, "groq", "quality")).toBe(1);
  expect(tensor.value(scope({ stage: "evaluate", consumer: "summary" }), "groq", "quality")).toBe(0.2);
  expect(tensor.value({}, "groq", "quality")).toBe(0.2);
});

test("context facets specialize preferences without pretending to own them", () => {
  const tensor = new Tensor(["workspace", "stage", "consumer"]);
  const writer = tensor.writer("workspace")!;
  writer.setValue("workspace", {}, "local", "cost", 0.2);
  writer.setValue("workspace", scope({ capability: "chat" }), "local", "cost", 0.8);

  expect(tensor.value(scope({ capability: "chat" }), "local", "cost")).toBe(0.8);
  expect(tensor.value(scope({ capability: "embedding" }), "local", "cost")).toBe(0.2);
});

test("scoped reductions do not mix ancestor branches", () => {
  const tensor = new Tensor(["session", "source", "device"]);
  const writer = tensor.writer("device")!;
  writer.setValue("device", scope({ source: "nzz", device: "A" }), "browser", "reliability", 1);
  writer.setValue("device", scope({ source: "nzz", device: "B" }), "browser", "reliability", 0);
  writer.setValue("device", scope({ source: "proton", device: "A" }), "browser", "reliability", 1);

  tensor.reduceMedian("device", "source", scope({ source: "nzz" }), "browser", "reliability");
  tensor.reduceMedian("device", "source", scope({ source: "proton" }), "browser", "reliability");
  expect(tensor.value(scope({ source: "nzz" }), "browser", "reliability")).toBe(0.5);
  expect(tensor.value(scope({ source: "proton" }), "browser", "reliability")).toBe(1);
});

test("corroboration counts distinct scoped reporters and excludes self-votes", () => {
  const tensor = new Tensor(["session", "device"]);
  for (const reporter of ["A", "B", "D"]) {
    tensor.writer("device")!.setValue("device", scope({ device: reporter }), "D", "can_scrape", 1);
  }
  tensor.corroborate("device", "session", {}, "D", "can_scrape", "device", 2);
  expect(tensor.value({}, "D", "can_scrape")).toBe(1);
});

test("documents are inert and scoped snapshots round-trip canonically", () => {
  const tensor = new Tensor(["workspace", "stage"]);
  const document = new Uint8Array([0, 1, 2, 255, 60, 62]);
  const workspace = scope({ workspace: "acme" });
  const writer = tensor.writer("workspace")!;
  writer.setBytes("workspace", workspace, "document", "state", document);
  writer.setValue("workspace", workspace, "groq", "latency", 0.2);
  writer.setWeight("workspace", workspace, "latency", 1);

  const json = tensor.toJson();
  expect(json).toContain('"b64":"AAEC/zw+"');
  const rebuilt = Tensor.fromJson(json)!;
  expect(rebuilt.bytes(workspace, "document", "state")).toEqual(document);
  expect(rebuilt.toJson()).toBe(json);
});

test("applyJson skips malformed scoped cells atomically", () => {
  const tensor = new Tensor(["global"]);
  tensor.applyJson(JSON.stringify([
    { layer: "global", scope: {}, option: "groq", dim: "latency", v: 0.1 },
    { layer: "global", scope: {}, option: "bad", dim: "latency", v: 0.5, w: true },
    { layer: "global", scope: {}, option: "cerebras", dim: "latency", v: 0.2 },
  ]));
  expect(tensor.value({}, "groq", "latency")).toBe(0.1);
  expect(tensor.value({}, "cerebras", "latency")).toBe(0.2);
  expect(tensor.value({}, "bad", "latency")).toBeUndefined();
});

test("clearScope removes exactly one preference set", () => {
  const tensor = new Tensor(["session", "job"]);
  const writer = tensor.writer("job")!;
  writer.setValue("job", scope({ job: "i" }), "browser", "reach", Infinity);
  writer.setValue("job", scope({ job: "j" }), "browser", "latency", 0.3);
  tensor.clearScope("job", scope({ job: "i" }));
  expect(tensor.value(scope({ job: "i" }), "browser", "reach")).toBeUndefined();
  expect(tensor.value(scope({ job: "j" }), "browser", "latency")).toBe(0.3);
});

test("non-finite values gate while explained resolve retains origins", () => {
  const tensor = new Tensor(["session", "source"]);
  const writer = tensor.writer("session")!;
  writer.setWeight("session", {}, "latency", 1);
  writer.setValue("session", {}, "keep", "latency", 0.2);
  writer.setValue("session", {}, "drop", "latency", 0.1);
  writer.setValue("source", scope({ source: "nzz" }), "drop", "reach", Infinity);

  const explained = tensor.resolveExplained(scope({ source: "nzz" }));
  expect(tensor.resolve(scope({ source: "nzz" }))).toEqual(["keep"]);
  expect(explained.find(({ option }) => option === "drop")?.viable).toBeFalse();
  expect(explained.find(({ option }) => option === "drop")?.dimensions.at(-1)?.value_origin.layer).toBe("source");
});
