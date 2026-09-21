// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden capability-gate vectors, shared with the Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/gate_vectors.rs`). Reading them here makes the vectors a
// cross-core DECISION-EQUIVALENCE guard for the gate primitive: both cores apply
// their own `gateCapabilities` / `gate_capabilities` to the identical base tensor and
// MUST drop the identical options and return the identical survivor order.
//
// Each vector carries a base `tensor`, a `gate` spec (writer level, the slice
// `(level, inst)`, the option ids, the required caps, and a serialized `offers` map
// standing in for the predicate), a read `cursor`, and the `expected` resolved order
// AFTER the gate is applied. The goldens were captured from the live `resolve` after
// the live gate — never hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gateCapabilities, Tensor, type Cursor } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/gate-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "gate-vectors");

interface GateVector {
  name: string;
  description?: string;
  tensor: { levels: string[]; cells: unknown[] };
  gate: {
    writerLevel: string;
    level: string;
    inst: string;
    options: string[];
    required: string[];
    offers: Record<string, string[]>;
  };
  cursor: Cursor;
  expected: string[];
}

function loadVectors(): GateVector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no gate vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as GateVector);
}

describe("golden gate vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      // Rebuild the base tensor from the persistence-port wire shape, exactly as the
      // Rust test does via Tensor::from_json.
      const t = Tensor.fromJson(JSON.stringify(vector.tensor));
      expect(t, `${vector.name}: tensor did not rebuild via fromJson`).toBeDefined();

      const { writerLevel, level, inst, options, required, offers } = vector.gate;
      const offersPred = (option: string, cap: string): boolean =>
        (offers[option] ?? []).includes(cap);

      const w = t!.writer(writerLevel);
      expect(w, `${vector.name}: no writer at ${writerLevel}`).toBeDefined();
      const e = gateCapabilities(w!, level, inst, options, required, offersPred);
      expect(e, `${vector.name}: gateCapabilities returned ${e}`).toBeUndefined();

      const resolved = t!.resolve(vector.cursor);
      // The TS gated resolve order MUST equal the golden (the Rust-captured order).
      expect(resolved).toEqual(vector.expected);
    });
  }
});
