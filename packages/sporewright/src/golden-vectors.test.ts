// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// REGRESSION GUARD — golden `resolve` vectors, shared with the Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/golden_vectors.rs`). Reading them here makes the
// vectors a cross-core DECISION-EQUIVALENCE guard: the TS `resolve` must produce
// the identical order the Rust core produced when the goldens were captured (the
// CONTRIBUTING.md "equivalence invariant"). A refactor that silently changes a
// routing decision on EITHER core breaks one of these two suites.
//
// The goldens were captured from the live Rust `resolve` — never hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Tensor, type Cursor } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "vectors");

interface Vector {
  name: string;
  description?: string;
  tensor: { layers: string[]; cells: unknown[] };
  cursor: Cursor;
  expected: string[];
}

function loadVectors(): Vector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no golden vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as Vector);
}

describe("golden resolve vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      // Rebuild the tensor from the persistence-port wire shape, exactly as the
      // Rust test does via Tensor::from_json.
      const t = Tensor.fromJson(JSON.stringify(vector.tensor));
      expect(t, `${vector.name}: tensor did not rebuild via fromJson`).toBeDefined();
      const resolved = t!.resolve(vector.cursor);
      // The TS core's resolve order MUST equal the golden (the Rust-captured order).
      expect(resolved).toEqual(vector.expected);
    });
  }
});
