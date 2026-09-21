// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { expect, test } from "bun:test";
import { Device, gateCapabilities, scope, Tensor } from "./index";

test("participant observations reduce through the scoped stack", () => {
  const tensor = new Tensor(["session", "source", "device"]);
  tensor.writer("session")!.setWeight("session", {}, "latency", 1);
  Device.at(tensor, "device", scope({ source: "nzz", device: "A" }))!
    .observe("browser", "latency", 0.2);
  Device.at(tensor, "device", scope({ source: "nzz", device: "B" }))!
    .observe("browser", "latency", 0.4);

  tensor.reduceAllMedian("device", "source", scope({ source: "nzz" }));
  expect(tensor.value(scope({ source: "nzz" }), "browser", "latency")).toBe(0.30000000000000004);
});

test("participant revision is layer-local feedback", () => {
  const tensor = new Tensor(["session", "source", "device"]);
  const participant = Device.at(tensor, "device", scope({ source: "nzz", device: "A" }))!;
  participant.observe("browser", "reliability", 0);
  const change = participant.revise("browser", "reliability", 1, 0.25);
  expect(typeof change).not.toBe("string");
  expect(tensor.value(scope({ source: "nzz", device: "A" }), "browser", "reliability")).toBe(0.25);
});

test("capability gates are scoped preferences", () => {
  const tensor = new Tensor(["session", "source", "job"]);
  const root = tensor.writer("session")!;
  root.setWeight("session", {}, "financial", 1);
  for (const option of ["full", "light", "curl"]) root.setValue("session", {}, option, "financial", 0);

  const job = scope({ source: "nzz", job: "j1" });
  const offers = (option: string, capability: string): boolean =>
    option === "full"
    || (option === "light" && capability === "browser:full")
    || (option === "curl" && capability === "ip:residential");
  expect(gateCapabilities(
    tensor.writer("job")!,
    "job",
    job,
    ["full", "light", "curl"],
    ["browser:full", "ip:residential"],
    offers,
  )).toBeUndefined();
  expect(tensor.resolve(job)).toEqual(["full"]);
  expect(tensor.resolve(scope({ source: "nzz", job: "j2" }))).toEqual(["curl", "full", "light"]);
});

test("reachability changes with the participant context", () => {
  const tensor = new Tensor(["session", "source", "device"]);
  const root = tensor.writer("session")!;
  root.setValue("session", {}, "browser", "latency", 0.2);
  root.setValue("session", {}, "native", "latency", 0.5);
  root.setWeight("session", {}, "latency", 1);

  const nzz = Device.at(tensor, "device", scope({ source: "nzz", device: "D" }))!;
  nzz.gate("browser", false);
  expect(tensor.resolve(scope({ source: "nzz", device: "D" }))).toEqual(["native"]);
  expect(tensor.resolve(scope({ source: "proton", device: "D" }))).toEqual(["browser", "native"]);
});
