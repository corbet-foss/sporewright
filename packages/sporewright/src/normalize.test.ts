// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// Unit tests for the per-dimension normalizer — decision-equivalent with the Rust
// `normalize.rs` unit tests. The cross-core golden parity lives in the shared
// `tests/norm-vectors/*.json` (norm-vectors.test.ts here + norm_vectors.rs on Rust).

import { expect, test } from "bun:test";
import { maxAbsScale, Norm, Tensor } from "./index";

test("effective weight is raw / scale", () => {
  const n = new Norm().withScale("financial", 0.01).withScale("latency", 5.0);
  expect(n.effectiveWeight("financial", 1.0)).toBe(100.0);
  expect(n.effectiveWeight("latency", 1.0)).toBe(0.2);
  // an unset dim is identity.
  expect(n.effectiveWeight("quality", 3.0)).toBe(3.0);
});

test("invalid scales fall back to identity", () => {
  const n = new Norm()
    .withScale("a", 0.0) // rejected (non-positive)
    .withScale("b", Infinity) // rejected (non-finite)
    .withScale("c", -2.0) // rejected (non-positive)
    .withScale("d", 4.0); // accepted
  expect(n.scale("a")).toBe(1.0);
  expect(n.scale("b")).toBe(1.0);
  expect(n.scale("c")).toBe(1.0);
  expect(n.scale("d")).toBe(4.0);
});

test("normalizeValue min-max and degenerate ranges", () => {
  expect(Norm.normalizeValue(5.0, 0.0, 10.0)).toBe(0.5);
  expect(Norm.normalizeValue(0.0, 0.0, 10.0)).toBe(0.0);
  expect(Norm.normalizeValue(10.0, 0.0, 10.0)).toBe(1.0);
  // clamp out-of-range.
  expect(Norm.normalizeValue(20.0, 0.0, 10.0)).toBe(1.0);
  expect(Norm.normalizeValue(-5.0, 0.0, 10.0)).toBe(0.0);
  // degenerate range → identity (no divide-by-zero).
  expect(Norm.normalizeValue(7.0, 3.0, 3.0)).toBe(7.0);
  expect(Norm.normalizeValue(7.0, 5.0, 1.0)).toBe(7.0);
  // a gate passes through untouched.
  expect(Norm.normalizeValue(Infinity, 0.0, 10.0)).toBe(Infinity);
  // OVERFLOW: a finite range that overflows makes (v-lo)/(hi-lo) = Infinity/Infinity =
  // NaN; we must fall back to identity, never leak a non-finite into a cell.
  const ov = Norm.normalizeValue(1e308, -1e308, 1e308);
  expect(Number.isFinite(ov)).toBe(true);
  expect(ov).toBe(1e308); // identity fallback
});

test("maxAbsScale picks the largest magnitude", () => {
  expect(maxAbsScale([0.2, -5.0, 3.0])).toBe(5.0);
  expect(maxAbsScale([0.0, 0.0])).toBe(1.0); // no magnitude → identity
  expect(maxAbsScale([Infinity, 2.0])).toBe(2.0); // gate ignored
  expect(maxAbsScale([])).toBe(1.0);
});

test("normalization flips a lexicographic order into a balanced one", () => {
  // Two options, two competing dims. financial lives on ~0.01, latency on ~5.
  // a: cheap but slow; b: pricey but fast.
  const t = new Tensor(["global"]);
  {
    const w = t.writer("global")!;
    w.setValue("global", {}, "a", "financial", 0.002);
    w.setValue("global", {}, "a", "latency", 5.0);
    w.setValue("global", {}, "b", "financial", 0.01);
    w.setValue("global", {}, "b", "latency", 0.5);
  }

  // RAW, equal weights: latency (~5) dwarfs financial (~0.01) → latency-dominated.
  {
    const w = t.writer("global")!;
    w.setWeight("global", {}, "financial", 1.0);
    w.setWeight("global", {}, "latency", 1.0);
  }
  expect(t.resolve({})).toEqual(["b", "a"]);

  // NORMALIZED, equal MRS weights: costs now commensurable.
  // a: 0.2 + 1.0 = 1.2 ; b: 1.0 + 0.1 = 1.1  → b leads but only just.
  const n = new Norm().withScale("financial", 0.01).withScale("latency", 5.0);
  {
    const w = t.writer("global")!;
    w.setWeight("global", {}, "financial", n.effectiveWeight("financial", 1.0));
    w.setWeight("global", {}, "latency", n.effectiveWeight("latency", 1.0));
  }
  expect(t.resolve({})).toEqual(["b", "a"]);

  // Crank financial's MRS weight: the cheap option `a` overtakes — impossible under
  // the raw latency-dominated sum at any reasonable financial weight.
  // a: 0.6 + 1.0 = 1.6 ; b: 3.0 + 0.1 = 3.1  → a wins.
  {
    const w = t.writer("global")!;
    w.setWeight("global", {}, "financial", n.effectiveWeight("financial", 3.0));
  }
  expect(t.resolve({})).toEqual(["a", "b"]);
});
