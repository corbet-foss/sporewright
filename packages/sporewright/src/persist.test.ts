// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { expect, test } from "bun:test";
import { MemStore, scope, Tensor } from "./index";

test("a snapshot survives a store", () => {
  const a = new Tensor(["global", "device"]);
  a.writer("device")!.setValue("device", scope({ device: "A" }), "groq", "latency", 0.1);
  const o = a.writer("global")!;
  o.setWeight("global", {}, "latency", 1.0);
  o.setBytes("global", {}, "doc:cv", "state", new Uint8Array([1, 2, 3]));

  const store = new MemStore();
  store.save(a.toJson());
  const b = Tensor.fromJson(store.load()!)!;

  expect(b.value({ device: "A" }, "groq", "latency")).toBe(0.1);
  expect(b.weight({}, "groq", "latency")).toBe(1.0);
  expect(b.bytes({}, "doc:cv", "state")).toEqual(new Uint8Array([1, 2, 3]));
});

test("fromJson rejects garbage", () => {
  expect(Tensor.fromJson("not json")).toBeUndefined();
  expect(Tensor.fromJson("{}")).toBeUndefined();
});

test("cross-core: a Rust-shaped snapshot restores in TS", () => {
  const snap = '{"layers":["global","device"],"cells":[{"layer":"device","scope":{"device":"A"},"option":"groq","dim":"latency","v":0.1}]}';
  const t = Tensor.fromJson(snap)!;
  expect(t.value({ device: "A" }, "groq", "latency")).toBe(0.1);
});
