// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden tensor-config vectors, shared with the Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/config_vectors.rs`). Each vector holds a declarative
// `config` (`{levels, seeds}`, where each seed is a reified Writer call) and the
// `expected` instantiated tensor wire (`{levels, cells}` — the `toJson` shape).
//
// Reading them here makes the vectors a cross-core DECISION-EQUIVALENCE guard for the
// declarative builder: the TS `instantiate` must produce the byte-identical tensor the
// Rust core produced (the faithfulness claim — "a config == those imperative Writer
// calls" — proven on BOTH cores at once). The `expected` came straight from the live
// Rust `instantiate`/`to_json`, never hand-fabricated.
//
// A seed's `value` uses the same wire codec as a cell's `v`: a finite number, an
// `"inf"`/`"-inf"`/`"nan"` sentinel for a gate, or a `{ b64 }` document.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  instantiate,
  Tensor,
  type Cursor,
  type SeedCell,
  type TensorConfig,
  type Value,
} from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/config-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "config-vectors");

interface WireSeed {
  floor: string;
  level: string;
  inst: string;
  option: string;
  dim: string;
  kind: SeedCell["kind"];
  value: number | string | { b64: string };
}
interface Vector {
  name: string;
  description?: string;
  config: { levels: string[]; seeds: WireSeed[] };
  cursor?: Cursor;
  resolve_expected?: string[];
  expected?: { levels: string[]; cells: unknown[] };
  /** Error-path vectors: instantiate must reject with this WriteError string. */
  expected_error?: "write-up" | "unknown-level" | "bad-value-kind";
}

function loadVectors(): Vector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no config vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as Vector);
}

function b64ToBytes(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Decode the seed `value` wire (same shape as a cell's `v`) into a {@link Value}. */
function decodeValue(v: WireSeed["value"]): Value {
  if (typeof v === "number") return { f64: v };
  if (typeof v === "string") {
    if (v === "inf") return { f64: Infinity };
    if (v === "-inf") return { f64: -Infinity };
    if (v === "nan") return { f64: NaN };
    throw new Error(`bad value sentinel: ${v}`);
  }
  return { bytes: b64ToBytes(v.b64) };
}

function parseConfig(c: Vector["config"]): TensorConfig {
  return {
    levels: c.levels,
    seeds: c.seeds.map((s) => ({
      floor: s.floor,
      level: s.level,
      inst: s.inst,
      option: s.option,
      dim: s.dim,
      kind: s.kind,
      value: decodeValue(s.value),
    })),
  };
}

describe("golden config vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      // Error-path vectors: instantiate must REJECT with the named WriteError string,
      // identically to Rust (an error rejection is a decision too).
      if (vector.expected_error) {
        expect(instantiate(parseConfig(vector.config))).toBe(vector.expected_error);
        return;
      }
      const t = instantiate(parseConfig(vector.config));
      expect(t, `${vector.name}: instantiate returned an error`).toBeInstanceOf(Tensor);

      // The instantiated tensor's wire must equal the golden the Rust core produced.
      const produced = JSON.parse((t as Tensor).toJson());
      expect(produced).toEqual(vector.expected);

      // If the vector pins a resolve order, the TS resolve must match it too.
      if (vector.resolve_expected) {
        expect((t as Tensor).resolve(vector.cursor ?? {})).toEqual(vector.resolve_expected);
      }
    });
  }
});
