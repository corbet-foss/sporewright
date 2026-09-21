// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * **Declarative tensor config** — a product declares its tensor as *data*.
 * Decision-equivalent with the Rust `config.rs`. See docs/MODEL.md §7 and
 * docs/ROUTING-MODEL.md §7.
 *
 * `MODEL.md` §7 / `ROUTING-MODEL.md` §7 draw the library boundary as **facts in →
 * decision out**: a product supplies its *levels*, *options*, *dims*, *weights*, and
 * *gates* (all data the engine routes over) and writes no routing logic. Today a
 * product expresses that seeding **imperatively** — a sequence of {@link Writer} calls
 * (e.g. a fleet seed, a cascade seed). This module lets the same seeding be a
 * plain {@link TensorConfig} value (levels + a list of seed cells) that
 * {@link instantiate} turns into a {@link Tensor}. The product declares its tensor as
 * DATA, not code — the SCOPE split made literal.
 *
 * **Faithful to the imperative path, not a parallel one.** A {@link SeedCell} is
 * exactly one {@link Writer} call: it carries a **`floor`** (the tier the minting
 * writer is granted) plus the write coordinate and the {@link SeedKind}.
 * {@link instantiate} mints a `writer(floor)` per cell and applies it, so **write-down
 * is enforced identically** — a config whose cell writes *coarser* than its `floor` is
 * rejected with the same `"write-up"` {@link WriteError} the imperative call would
 * raise, and an unknown level with `"unknown-level"`. There is no second write path: a
 * config IS a list of `Writer` calls, just reified as data. Expressing an existing
 * imperative seeding as a `TensorConfig` and instantiating it yields a tensor
 * **byte-identical** to the imperatively-built one (the round-trip test, mirrored in
 * Rust).
 *
 * **No engine change, decision-equivalent.** `instantiate` only *writes* cells the
 * `Writer` already writes; `resolve`/`reduce`/`fold` are untouched. The Rust and TS
 * builders apply cells in the **same declared order**, so the resulting tensors are
 * identical and any later `resolve` is decision-equivalent. Gates need no special kind:
 * a gate is a non-finite **value** (a `SeedCell` value of `±∞`/`NaN`), so it seeds
 * through the ordinary value path and `resolve` drops it exactly as before.
 */
import type { Value, WriteError } from "./tensor";
import { SHARED, Tensor } from "./tensor";

/** Which layer a {@link SeedCell} writes — the data twin of the (internal) write kind.
 *  Mirrors the four {@link Writer} setters: a scalar value, a document blob, an
 *  option-shared weight, or a per-option (gas-pedal) weight. */
export type SeedKind = "value" | "bytes" | "weight" | "option-weight";

/** One declarative seed — exactly one {@link Writer} call reified as data.
 *
 *  `floor` is the tier the minting writer is granted (the write-down floor); `level` is
 *  the tier the cell lands at (must be `floor` or finer). For `kind: "weight"` the
 *  `option` is ignored (the shared `""` slot is always used). The `value` is a
 *  {@link Value} so a gate (`{ f64: Infinity }`) and a document (`{ bytes }`) both
 *  express naturally. */
export interface SeedCell {
  /** The write-down floor — the tier the writer minted for this cell is granted. */
  floor: string;
  /** The tier the cell lands at (must be `floor` or finer, else `"write-up"`). */
  level: string;
  /** The instance on `level` (the shared `""` slot, or a concrete instance). */
  inst: string;
  /** The option this cell is about (ignored for `kind: "weight"`). */
  option: string;
  /** The dimension this cell is about. */
  dim: string;
  /** Which layer to write. */
  kind: SeedKind;
  /** The payload — a scalar (incl. a non-finite gate) or document bytes. */
  value: Value;
}

/** A scalar **value** seed (`Writer.setValue`). A non-finite `v` is a gate. */
export function valueCell(floor: string, level: string, inst: string, option: string, dim: string, v: number): SeedCell {
  return { floor, level, inst, option, dim, kind: "value", value: { f64: v } };
}

/** A **document** seed (`Writer.setBytes`). */
export function bytesCell(floor: string, level: string, inst: string, option: string, dim: string, b: Uint8Array): SeedCell {
  return { floor, level, inst, option, dim, kind: "bytes", value: { bytes: b } };
}

/** An **option-shared weight** seed (`Writer.setWeight`). `option` is the shared `""`
 *  slot by construction. */
export function weightCell(floor: string, level: string, inst: string, dim: string, w: number): SeedCell {
  return { floor, level, inst, option: SHARED, dim, kind: "weight", value: { f64: w } };
}

/** A **per-option weight** seed — the gas pedal (`Writer.setOptionWeight`). */
export function optionWeightCell(floor: string, level: string, inst: string, option: string, dim: string, w: number): SeedCell {
  return { floor, level, inst, option, dim, kind: "option-weight", value: { f64: w } };
}

/** A product's tensor declared as **data**: the ordered context-tier skeleton plus the
 *  seed cells, applied in declared order. Plain data — no `Writer`, no engine state. */
export interface TensorConfig {
  /** The context-tier skeleton, coarsest first — exactly the `Tensor` constructor arg. */
  levels: string[];
  /** The seed cells, applied **in this order** (so a later cell overwrites an earlier
   *  one at the same coordinate, exactly as repeated imperative writes would). */
  seeds: SeedCell[];
}

/** Build a {@link TensorConfig} from levels + an optional initial seed list. Push more
 *  with {@link withCell} — order is preserved (load-bearing for overwrite semantics and
 *  cross-core equivalence). */
export function tensorConfig(levels: string[], seeds: SeedCell[] = []): TensorConfig {
  return { levels: [...levels], seeds: [...seeds] };
}

/** Append a seed cell to a config (returns a new config; order preserved). */
export function withCell(config: TensorConfig, cell: SeedCell): TensorConfig {
  return { levels: config.levels, seeds: [...config.seeds, cell] };
}

function asF64(v: Value): number | undefined {
  return "f64" in v ? v.f64 : undefined;
}
function asBytes(v: Value): Uint8Array | undefined {
  return "bytes" in v ? v.bytes : undefined;
}

/** Build a {@link Tensor} from a {@link TensorConfig}: create the tier skeleton, then
 *  for each seed mint a `writer(floor)` and apply the cell — the **same
 *  write-down-enforced path** the imperative seeding takes. Cells are applied in
 *  declared order.
 *
 *  Returns the {@link Tensor} on success (byte-identical to the same sequence of
 *  imperative `Writer` calls), or the first {@link WriteError} — one of three rejection
 *  causes: `"unknown-level"` (an unknown `floor`/`level`), `"write-up"` (a cell coarser
 *  than its `floor`), or `"bad-value-kind"` (a payload whose declared kind disagrees with
 *  its value type, e.g. a `"bytes"` cell carrying an `{ f64 }`). */
export function instantiate(config: TensorConfig): Tensor | WriteError {
  const t = new Tensor(config.levels);
  for (const cell of config.seeds) {
    // Mint a writer granted at the cell's floor — an unknown floor is the same error
    // the imperative `t.writer(floor)` would surface (undefined → "unknown-level").
    const w = t.writer(cell.floor);
    if (w === undefined) return "unknown-level";
    let e: WriteError | undefined;
    // A kind/value-payload mismatch (e.g. `kind:"bytes"` carrying an `{ f64 }`, or a
    // scalar kind carrying `{ bytes }`) is a malformed CELL, not an unknown TIER — return
    // "bad-value-kind" so the author is pointed at the cell, not the levels list.
    switch (cell.kind) {
      case "value": {
        const v = asF64(cell.value);
        if (v === undefined) return "bad-value-kind";
        e = w.setValue(cell.level, cell.inst, cell.option, cell.dim, v);
        break;
      }
      case "bytes": {
        const b = asBytes(cell.value);
        if (b === undefined) return "bad-value-kind";
        e = w.setBytes(cell.level, cell.inst, cell.option, cell.dim, b);
        break;
      }
      case "weight": {
        const v = asF64(cell.value);
        if (v === undefined) return "bad-value-kind";
        e = w.setWeight(cell.level, cell.inst, cell.dim, v);
        break;
      }
      case "option-weight": {
        const v = asF64(cell.value);
        if (v === undefined) return "bad-value-kind";
        e = w.setOptionWeight(cell.level, cell.inst, cell.option, cell.dim, v);
        break;
      }
    }
    if (e) return e;
  }
  return t;
}
