// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AddressedField, type Address } from "./field";

interface Declaration {
  kind: "prior" | "weight" | "gate";
  address: Address;
  option: string;
  dimension: string;
  value: number;
}

interface Observation {
  address: Address;
  option: string;
  dimension: string;
  value: number;
  variance: number;
}

interface Maintenance {
  kind: "compact" | "discount";
  address: Address;
  option: string;
  dimension: string;
  factor?: number;
}

interface Query {
  address: Address;
  temperature: number;
  allowed?: string[];
  expected: string[];
  means: Record<string, Record<string, number>>;
}

interface Vector {
  name: string;
  layers: string[];
  declarations: Declaration[];
  observations: Observation[];
  maintenance?: Maintenance[];
  queries: Query[];
}

const here = dirname(fileURLToPath(import.meta.url));
const vectorsDir = join(here, "..", "..", "..", "tests", "field-vectors");
const vectors = readdirSync(vectorsDir)
  .filter((file) => file.endsWith(".json"))
  .sort()
  .map((file) => JSON.parse(readFileSync(join(vectorsDir, file), "utf8")) as Vector);

describe("addressed field vectors — Rust/TypeScript decision equivalence", () => {
  for (const vector of vectors) {
    it(vector.name, () => {
      const field = new AddressedField(vector.layers);
      const writer = field.rootWriter();
      for (const declaration of vector.declarations) {
        const error = declaration.kind === "prior"
          ? writer.setPrior(declaration.address, declaration.option, declaration.dimension, declaration.value)
          : declaration.kind === "weight"
            ? writer.setOptionWeight(declaration.address, declaration.option, declaration.dimension, declaration.value)
            : writer.setGate(declaration.address, declaration.option, declaration.dimension, declaration.value > 0);
        expect(error).toBeUndefined();
      }
      for (const observation of vector.observations) {
        expect(writer.observe(
          observation.address,
          observation.option,
          observation.dimension,
          observation.value,
          observation.variance,
        )).not.toBeString();
      }
      for (const maintenance of vector.maintenance ?? []) {
        const trace = maintenance.kind === "compact"
          ? writer.compactSubtree(maintenance.address, maintenance.option, maintenance.dimension)
          : writer.discountEvidence(
            maintenance.address,
            maintenance.option,
            maintenance.dimension,
            maintenance.factor!,
          );
        expect(trace).not.toBeString();
      }
      for (const query of vector.queries) {
        const decision = query.allowed === undefined
          ? field.decide(query.address, { temperature: query.temperature })
          : field.decideAmong(query.address, query.allowed, { temperature: query.temperature });
        if (typeof decision === "string") throw new Error(decision);
        expect(decision.alternatives.filter(({ viable }) => viable).map(({ option }) => option)).toEqual(query.expected);
        for (const [option, dimensions] of Object.entries(query.means)) {
          const candidate = decision.alternatives.find((value) => value.option === option)!;
          for (const [dimension, expected] of Object.entries(dimensions)) {
            const actual = candidate.dimensions.find((value) => value.dimension === dimension)!.mean;
            expect(Math.abs(actual - expected)).toBeLessThan(1e-10);
          }
        }
      }
    });
  }
});
