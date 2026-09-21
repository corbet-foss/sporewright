// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden corroborate vectors, shared with the Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/corroborate_vectors.rs`). Both cores count independent
// reporters with the identical self-exclusion (inst !== option) and `> 0` attestation
// threshold and MUST materialize the identical subjects at the identical quorum
// boundary. Before this corpus, `corroborate` was pinned only by independent per-core
// unit tests — a one-sided change would pass both suites while diverging the
// corroboration decision. This pins the faker-resistant count (named in the
// CONTRIBUTING equivalence invariant) as a cross-core decision-equivalent capability.
//
// Each vector carries a `feeder` spec (levels, fromLevel/toLevel/toInst, quorum, and
// raw per-instance attestation `writes`), the exact `materialized` subjects that must
// exist after corroborate, and the `absent` subjects that must NOT. The goldens were
// captured from the live corroborate — never hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Tensor, type Cursor } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/corroborate-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "corroborate-vectors");

interface CorroborateVector {
  name: string;
  kind: "corroborate";
  description?: string;
  feeder: {
    levels: string[];
    fromLevel: string;
    toLevel: string;
    toInst: string;
    quorum: number;
    writes: { inst: string; option: string; dim: string; value: number }[];
  };
  cursor: Cursor;
  materialized: { option: string; dim: string; value: number }[];
  absent: { option: string; dim: string }[];
}

function loadVectors(): CorroborateVector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no corroborate vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as CorroborateVector);
}

describe("golden corroborate vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      expect(vector.kind, `${vector.name}: only the 'corroborate' kind is implemented`).toBe("corroborate");

      const t = new Tensor(vector.feeder.levels);
      for (const { inst, option, dim, value } of vector.feeder.writes) {
        const w = t.writer(vector.feeder.fromLevel);
        expect(w, `${vector.name}: no feeder writer at ${vector.feeder.fromLevel}`).toBeDefined();
        const e = w!.setValue(vector.feeder.fromLevel, inst, option, dim, value);
        expect(e, `${vector.name}: setValue returned ${e}`).toBeUndefined();
      }

      t.corroborate(vector.feeder.fromLevel, vector.feeder.toLevel, vector.feeder.toInst, vector.feeder.quorum);

      for (const { option, dim, value } of vector.materialized) {
        expect(t.value(vector.cursor, option, dim)).toBe(value);
      }
      for (const { option, dim } of vector.absent) {
        expect(t.value(vector.cursor, option, dim)).toBeUndefined();
      }
    });
  }
});
