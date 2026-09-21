// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden normalization vectors, shared with the Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/norm_vectors.rs`). Both cores build a `Norm` from the
// per-dim `scales`, write each dim's EFFECTIVE weight (raw MRS / scale) through the
// UNCHANGED `resolve`, and MUST return the identical order. This pins the
// normalizer as a cross-core decision-equivalent capability: the units fix is
// byte-for-byte the same routing decision on TS and Rust.
//
// Each weight-side vector carries a base `tensor` of RAW value cells, a `norm` spec
// (writeLevel, a per-dim `scales` map, the per-dim raw `weights` to derive), a read
// `cursor`, and the `expected` resolved order AFTER the effective weights are written.
//
// A vector may instead carry a `valueNorm` block (the VALUE-side route): the runner
// rewrites each raw `(option, dim)` value as `Norm.normalizeValue(raw, lo, hi)` (per-dim
// `ranges`), writes the shared `weights`, resolves, and asserts BOTH the per-cell
// `expectedValues` (bit-exact), the `expectedGated` options (dropped by resolve), AND
// the `expected` order. This is the only vector that drives the value-side min-max
// route the docs call "decision-equivalent across cores".
//
// The goldens were captured from the live normalized `resolve` — never hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Norm, Tensor, type Cursor } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/norm-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "norm-vectors");

interface NormVector {
  name: string;
  description?: string;
  tensor: { levels: string[]; cells: unknown[] };
  norm?: {
    writeLevel: string;
    scales: Record<string, number>;
    weights: Record<string, number>;
  };
  valueNorm?: {
    writeLevel: string;
    ranges: Record<string, { lo: number; hi: number }>;
    weights: Record<string, number>;
  };
  expectedValues?: { option: string; dim: string; value: number }[];
  expectedGated?: { option: string; dim: string }[];
  cursor: Cursor;
  expected: string[];
}

function loadVectors(): NormVector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no norm vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as NormVector);
}

describe("golden norm vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      // Rebuild the base tensor from the persistence-port wire shape, exactly as the
      // Rust test does via Tensor::from_json.
      const t = Tensor.fromJson(JSON.stringify(vector.tensor));
      expect(t, `${vector.name}: tensor did not rebuild via fromJson`).toBeDefined();

      if (vector.valueNorm) {
        // VALUE-side route: rewrite each raw value to its normalized value (a
        // non-finite gate passes through untouched), write the shared weights, resolve.
        const { writeLevel, ranges, weights } = vector.valueNorm;
        const options = t!.options(vector.cursor);
        // Collect normalized writes before opening a writer (mirrors the Rust harness).
        const normWrites: { option: string; dim: string; value: number }[] = [];
        for (const [dim, { lo, hi }] of Object.entries(ranges)) {
          for (const opt of options) {
            const raw = t!.value(vector.cursor, opt, dim);
            if (raw !== undefined) normWrites.push({ option: opt, dim, value: Norm.normalizeValue(raw, lo, hi) });
          }
        }
        const w = t!.writer(writeLevel);
        expect(w, `${vector.name}: no writer at ${writeLevel}`).toBeDefined();
        for (const { option, dim, value } of normWrites) {
          const e = w!.setValue(writeLevel, "", option, dim, value);
          expect(e, `${vector.name}: setValue(${option},${dim}) returned ${e}`).toBeUndefined();
        }
        for (const [dim, raw] of Object.entries(weights)) {
          const e = w!.setWeight(writeLevel, "", dim, raw);
          expect(e, `${vector.name}: setWeight(${dim}) returned ${e}`).toBeUndefined();
        }

        for (const { option, dim, value } of vector.expectedValues ?? []) {
          expect(t!.value(vector.cursor, option, dim)).toBe(value);
        }
        const resolved = t!.resolve(vector.cursor);
        for (const { option } of vector.expectedGated ?? []) {
          expect(resolved).not.toContain(option);
        }
        expect(resolved).toEqual(vector.expected);
        return;
      }

      const { writeLevel, scales, weights } = vector.norm!;
      const n = new Norm();
      for (const [dim, s] of Object.entries(scales)) n.withScale(dim, s);

      const w = t!.writer(writeLevel);
      expect(w, `${vector.name}: no writer at ${writeLevel}`).toBeDefined();
      for (const [dim, raw] of Object.entries(weights)) {
        const e = w!.setWeight(writeLevel, "", dim, n.effectiveWeight(dim, raw));
        expect(e, `${vector.name}: setWeight(${dim}) returned ${e}`).toBeUndefined();
      }

      const resolved = t!.resolve(vector.cursor);
      // The TS normalized resolve order MUST equal the golden (the Rust-captured order).
      expect(resolved).toEqual(vector.expected);
    });
  }
});
