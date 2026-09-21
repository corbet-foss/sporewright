// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
// Shared with `crates/sporewright/tests/trust_vectors.rs`.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { updatePairTrust } from "./trust";

interface Vector {
  name: string;
  incomingTrust: number;
  currentTrust: number;
  agree: boolean;
  freshness: number;
  sourceVolatility: number;
  expectedIncomingTrust: number;
  expectedCurrentTrust: number;
}

const here = dirname(fileURLToPath(import.meta.url));
const vectorsDir = join(here, "..", "..", "..", "tests", "trust-vectors");

describe("golden trust vectors — cross-core decision equivalence", () => {
  const files = readdirSync(vectorsDir).filter((file) => file.endsWith(".json")).sort();
  if (files.length === 0) throw new Error(`no trust vectors in ${vectorsDir}`);
  for (const file of files) {
    const vector = JSON.parse(readFileSync(join(vectorsDir, file), "utf8")) as Vector;
    it(`pins ${vector.name}`, () => {
      const result = updatePairTrust(vector);
      expect(result.incomingTrust).toBeCloseTo(vector.expectedIncomingTrust, 12);
      expect(result.currentTrust).toBeCloseTo(vector.expectedCurrentTrust, 12);
    });
  }
});
