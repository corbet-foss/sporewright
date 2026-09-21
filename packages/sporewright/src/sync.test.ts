// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { expect, test } from "bun:test";
import { Device, gateCapabilities, Tensor } from "./index";

const at = (device: string) => ({ device });

test("the device-mesh loop over primitives (byte-equivalent median)", () => {
  const t = new Tensor(["global", "device"]);
  t.writer("global")!.setWeight("global", "", "latency", 1.0);
  Device.at(t, "device", "A")!.observe("groq", "latency", 0.2);
  const b = Device.at(t, "device", "B")!;
  b.observe("groq", "latency", 0.4);
  b.gate("peer:C", false);

  t.rollUp("device", "global", "");
  expect(t.value({}, "groq", "latency")).toBe(0.30000000000000004);
  expect(t.resolve(at("B"))).toEqual(["groq"]); // unreachable peer dropped
});

test("devices attest and the orchestrator corroborates", () => {
  const t = new Tensor(["global", "device"]);
  for (const r of ["A", "B"]) Device.at(t, "device", r)!.attest("peer:D", "can_scrape", true);
  t.corroborate("device", "global", "", 2);
  expect(t.value({}, "peer:D", "can_scrape")).toBe(1.0);
});

test("gateCapabilities drops options missing a required cap", () => {
  // Mirrors the Rust unit test: three options, a job requires two caps; only the
  // option offering BOTH survives resolve at the job slice.
  const t = new Tensor(["global", "source", "job"]);
  {
    const w = t.writer("global")!;
    w.setWeight("global", "", "financial", 1.0);
    for (const o of ["full", "light", "curl"]) w.setValue("global", "", o, "financial", 0.0);
  }
  const offers = (o: string, c: string): boolean =>
    (o === "full" && c === "browser:full") ||
    (o === "full" && c === "ip:residential") ||
    (o === "light" && c === "browser:full") ||
    (o === "curl" && c === "ip:residential");
  {
    const w = t.writer("global")!;
    const e = gateCapabilities(w, "job", "j1", ["full", "light", "curl"], ["browser:full", "ip:residential"], offers);
    expect(e).toBeUndefined();
  }
  expect(t.resolve({ job: "j1" })).toEqual(["full"]);
  // Per-slice: a different job with no gates keeps the whole fleet (by name, all 0.0).
  expect(t.resolve({ job: "j2" })).toEqual(["curl", "full", "light"]);
});

test("gateCapabilities with no required caps gates nothing", () => {
  const t = new Tensor(["global", "job"]);
  {
    const w = t.writer("global")!;
    w.setWeight("global", "", "financial", 1.0);
    w.setValue("global", "", "a", "financial", 0.0);
    w.setValue("global", "", "b", "financial", 0.0);
  }
  const w = t.writer("global")!;
  // offers() is never consulted when there are no required caps.
  expect(gateCapabilities(w, "job", "j", ["a", "b"], [], () => false)).toBeUndefined();
  expect(t.resolve({ job: "j" })).toEqual(["a", "b"]);
});

test("gateCapabilities refuses to write up", () => {
  const t = new Tensor(["global", "job"]);
  const w = t.writer("job")!; // floored at "job", cannot gate the coarser "global"
  expect(gateCapabilities(w, "global", "", ["a"], ["x"], () => false)).toBe("write-up");
});

test("peer is an option gated by reachability", () => {
  const t = new Tensor(["global", "device"]);
  const o = t.writer("global")!;
  o.setValue("global", "", "local", "latency", 0.5);
  o.setValue("global", "", "peer:A", "latency", 0.2);
  o.setWeight("global", "", "latency", 1.0);

  Device.at(t, "device", "D")!.gate("peer:A", false);
  expect(t.resolve(at("D"))).toEqual(["local"]);
  Device.at(t, "device", "D")!.gate("peer:A", true);
  expect(t.resolve(at("D"))).toEqual(["peer:A", "local"]);
});
