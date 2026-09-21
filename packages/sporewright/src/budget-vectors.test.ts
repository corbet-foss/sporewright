// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// CROSS-CORE GUARD — golden budget/price + token-bucket vectors, shared with the
// Rust core.
//
// These are the SAME JSON files asserted by the Rust integration test
// (`crates/sporewright/tests/budget_vectors.rs`). Both cores build the identical
// `Budget` / `TokenBucket`, drive the identical step sequence, and MUST reach the
// identical resolved order, the identical final prices `λ`, and the identical admit
// decisions. This pins the budget layer as a cross-core decision-equivalent
// capability.
//
// Three vector kinds:
//  - kind="price": a base `tensor`, a `budget` spec (step, priceLevel, pools,
//    option→pool), a sequence of `ticks` (each a list of {pool, usage}); after every
//    tick the prices are published as the SHARED budget-dim weight and the tensor is
//    resolved. Asserts `expectedOrder` and `expectedLambda`. The `ticks` feed `measured`
//    DIRECTLY, bypassing the raw-usage→reduce_sum→aggregate_usage feeder.
//  - kind="usage": exercises the FEEDER path the `price` kind bypasses. A `feeder` spec
//    lists raw per-instance usage writes `{option, inst, usage}` in a SPECIFIC order
//    (insertion order is load-bearing — it must differ from the canonical sorted order so
//    the sum-order fix is actually under test). The runner builds a fresh feeder tensor,
//    writes the usage in the given order, runs `aggregateUsage` (which internally calls
//    `reduceSum` per option then sums per-option totals per pool), then `tick`→publish→
//    resolve on the base tensor. Asserts `expectedMeasured` (the exact per-pool sums —
//    a bit-for-bit cross-core check), `expectedLambda`, and `expectedOrder`. This is the
//    only kind that guards `reduceSum`'s instance-sum order AND `aggregateUsage`'s
//    option-sum order, both of which diverge across cores unless summed canonically.
//  - kind="bucket": a `bucket` spec (capacity, refillPerSec, start) and a list of
//    `admits` times; asserts the `expectedAdmits` booleans.
//
// The goldens were captured from the live budget layer on both cores — never
// hand-fabricated.

import { describe, expect, it } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Budget, scope, Tensor, TokenBucket, type Cursor, type Scope } from "./index";

const here = dirname(fileURLToPath(import.meta.url));
// src → packages/sporewright → packages → sporewright(root) → tests/budget-vectors
const VECTORS_DIR = join(here, "..", "..", "..", "tests", "budget-vectors");

interface PriceVector {
  name: string;
  kind: "price";
  description?: string;
  tensor: { layers: string[]; cells: unknown[] };
  budget: {
    step: number;
    priceLevel: string;
    pools: { pool: string; cap: number }[];
    options: { option: string; pool: string }[];
  };
  ticks: { pool: string; usage: number }[][];
  cursor: Cursor;
  expectedOrder: string[];
  expectedLambda: Record<string, number>;
}

interface UsageVector {
  name: string;
  kind: "usage";
  description?: string;
  tensor: { layers: string[]; cells: unknown[] };
  budget: {
    step: number;
    priceLevel: string;
    pools: { pool: string; cap: number }[];
    options: { option: string; pool: string }[];
  };
  // Raw per-instance usage writes, in load-bearing INSERTION order.
  feeder: {
    levels: string[];
    fromLevel: string;
    toLevel: string;
    toInst: string;
    writes: { option: string; inst: string; usage: number }[];
  };
  cursor: Cursor;
  expectedMeasured: Record<string, number>;
  expectedLambda: Record<string, number>;
  expectedOrder: string[];
}

interface BucketVector {
  name: string;
  kind: "bucket";
  description?: string;
  bucket: { capacity: number; refillPerSec: number; start: number };
  admits: number[];
  expectedAdmits: boolean[];
}

type Vector = PriceVector | UsageVector | BucketVector;

function loadVectors(): Vector[] {
  const files = readdirSync(VECTORS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  if (files.length === 0) throw new Error(`no budget vectors in ${VECTORS_DIR}`);
  return files.map((f) => JSON.parse(readFileSync(join(VECTORS_DIR, f), "utf8")) as Vector);
}

function buildBudget(spec: PriceVector["budget"] | UsageVector["budget"]): Budget {
  const b = new Budget().withStep(spec.step);
  for (const { pool, cap } of spec.pools) b.withPool(pool, cap);
  for (const { option, pool } of spec.options) b.mapOption(option, pool);
  return b;
}

describe("golden budget vectors — cross-core decision equivalence (shared with Rust)", () => {
  for (const vector of loadVectors()) {
    it(`pins ${vector.name}`, () => {
      if (vector.kind === "price") {
        const t = Tensor.fromJson(JSON.stringify(vector.tensor));
        expect(t, `${vector.name}: tensor did not rebuild via fromJson`).toBeDefined();
        const budget = buildBudget(vector.budget);
        const level = vector.budget.priceLevel;

        for (const tick of vector.ticks) {
          const measured = new Map<string, number>();
          for (const { pool, usage } of tick) measured.set(pool, usage);
          budget.tick(measured);
          const w = t!.writer(level);
          expect(w, `${vector.name}: no writer at ${level}`).toBeDefined();
          const e = budget.publishPrices(w!, level, {});
          expect(e, `${vector.name}: publishPrices returned ${e}`).toBeUndefined();
        }
        // If there were no ticks at all, still publish the (zero) prices once.
        if (vector.ticks.length === 0) {
          budget.publishPrices(t!.writer(level)!, level, {});
        }

        const resolved = t!.resolve(vector.cursor);
        expect(resolved).toEqual(vector.expectedOrder);
        for (const [pool, lam] of Object.entries(vector.expectedLambda)) {
          expect(budget.lambda(pool)).toBe(lam);
        }
      } else if (vector.kind === "usage") {
        const t = Tensor.fromJson(JSON.stringify(vector.tensor));
        expect(t, `${vector.name}: tensor did not rebuild via fromJson`).toBeDefined();
        const budget = buildBudget(vector.budget);

        // Build a FRESH feeder tensor and write the raw per-instance usage in the
        // vector's INSERTION order (load-bearing: it must differ from the canonical
        // sorted order so the sum-order fix is under test).
        const feeder = Tensor.fromJson(
          JSON.stringify({ layers: vector.feeder.levels, cells: [] }),
        );
        expect(feeder, `${vector.name}: feeder did not build`).toBeDefined();
        for (const { option, inst, usage } of vector.feeder.writes) {
          const w = feeder!.writer(vector.feeder.fromLevel);
          expect(w, `${vector.name}: no feeder writer at ${vector.feeder.fromLevel}`).toBeDefined();
          const cellScope: Scope = vector.feeder.toInst === ""
            ? scope({ [vector.feeder.fromLevel]: inst })
            : scope({
              [vector.feeder.toLevel]: vector.feeder.toInst,
              [vector.feeder.fromLevel]: inst,
            });
          const r = budget.writeUsage(w!, vector.feeder.fromLevel, cellScope, option, usage);
          expect(typeof r === "boolean" ? true : false, `${vector.name}: writeUsage error ${r}`).toBe(true);
        }

        // aggregateUsage internally reduceSums per option then sums per-option totals
        // per pool — both summations must be canonical for cross-core equivalence.
        const measured = budget.aggregateUsage(
          feeder!,
          vector.feeder.fromLevel,
          vector.feeder.toLevel,
          vector.feeder.toInst === "" ? {} : scope({ [vector.feeder.toLevel]: vector.feeder.toInst }),
        );
        // Exact (bit-for-bit) per-pool measured sums — the primary divergence guard.
        for (const [pool, want] of Object.entries(vector.expectedMeasured)) {
          expect(measured.get(pool)).toBe(want);
        }

        budget.tick(measured);
        const w = t!.writer(vector.budget.priceLevel);
        expect(w, `${vector.name}: no writer at ${vector.budget.priceLevel}`).toBeDefined();
        const e = budget.publishPrices(w!, vector.budget.priceLevel, {});
        expect(e, `${vector.name}: publishPrices returned ${e}`).toBeUndefined();

        for (const [pool, lam] of Object.entries(vector.expectedLambda)) {
          expect(budget.lambda(pool)).toBe(lam);
        }
        const resolved = t!.resolve(vector.cursor);
        expect(resolved).toEqual(vector.expectedOrder);
      } else {
        const { capacity, refillPerSec, start } = vector.bucket;
        const tb = new TokenBucket(capacity, refillPerSec, start);
        const got = vector.admits.map((now) => tb.admit(now));
        expect(got).toEqual(vector.expectedAdmits);
      }
    });
  }
});
