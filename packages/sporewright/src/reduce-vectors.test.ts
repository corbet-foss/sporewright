// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden reduce_median / roll_up vectors, shared with the Rust
// core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/reduce_vectors.rs`). Both cores roll the identical raw
// per-instance value cells up via `rollUp` (`reduceMedian` for every present
// `(option, dim)`) and MUST materialize the identical median. Before this corpus,
// `reduceMedian` was pinned only by independent per-core unit literals (both hardcode
// 0.30000000000000004) that could drift apart silently. This pins the certified
// integrity up-path (MODEL.md §3) as a cross-core decision-equivalent capability.
//
// Each vector carries a `feeder` spec (levels, fromLevel/toLevel/toInst, and raw
// per-instance `writes` in a SCRAMBLED order so the value-sort inside reduceMedian is
// under test). A `value` of the string "inf"/"-inf"/"nan" writes the matching
// non-finite sentinel (exercising the finite filter). The goldens were captured from
// the live reduceMedian — never hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { scope, Tensor, type Cursor, type Scope } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/reduce-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "reduce-vectors");

interface ReduceVector {
  name: string;
  kind: "median";
  description?: string;
  feeder: {
    levels: string[];
    fromLevel: string;
    toLevel: string;
    toInst: string;
    writes: { inst: string; option: string; dim: string; value: number | string }[];
  };
  cursor: Cursor;
  expected: { option: string; dim: string; value: number }[];
}

function loadVectors(): ReduceVector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no reduce vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as ReduceVector);
}

// Honor the wire sentinels for non-finite values so a `"value": "inf"` write
// exercises the finite filter, matching the Rust harness.
function parseValue(v: number | string): number {
  if (typeof v === "number") return v;
  if (v === "inf") return Infinity;
  if (v === "-inf") return -Infinity;
  if (v === "nan") return NaN;
  throw new Error(`unrecognized value sentinel ${JSON.stringify(v)}`);
}

describe("golden reduce vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      expect(vector.kind, `${vector.name}: only the 'median' kind is implemented`).toBe("median");

      // Build a FRESH feeder tensor and write the raw per-instance values in the
      // vector's SCRAMBLED order — reduceMedian sorts before indexing, so the result
      // must be independent of this write order.
      const t = new Tensor(vector.feeder.levels);
      for (const { inst, option, dim, value } of vector.feeder.writes) {
        const w = t.writer(vector.feeder.fromLevel);
        expect(w, `${vector.name}: no feeder writer at ${vector.feeder.fromLevel}`).toBeDefined();
        const cellScope: Scope = vector.feeder.toInst === ""
          ? scope({ [vector.feeder.fromLevel]: inst })
          : scope({
            [vector.feeder.toLevel]: vector.feeder.toInst,
            [vector.feeder.fromLevel]: inst,
          });
        const e = w!.setValue(vector.feeder.fromLevel, cellScope, option, dim, parseValue(value));
        expect(e, `${vector.name}: setValue returned ${e}`).toBeUndefined();
      }

      const targetScope = vector.feeder.toInst === ""
        ? {}
        : scope({ [vector.feeder.toLevel]: vector.feeder.toInst });
      const reduction = t.reduceAllMedian(vector.feeder.fromLevel, vector.feeder.toLevel, targetScope);
      expect(typeof reduction, `${vector.name}: reduction returned ${reduction}`).not.toBe("string");

      for (const { option, dim, value } of vector.expected) {
        expect(t.value(vector.cursor, option, dim)).toBe(value);
      }
    });
  }
});
