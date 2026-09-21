// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * The one sparse tensor of facts — the heart of the wright. Byte-equivalent with
 * the Rust `tensor.rs`. See docs/MODEL.md.
 *
 * A cell is a fact at a coordinate `(level, instance, option, dimension)` holding
 * two co-located layers: a value (the learned truth, `f64`|`bytes`) and a weight
 * (the governing will, `f64`). Reading is a cursor; the fold takes the
 * most-specific set layer. `resolve` is Σ value·weight.
 *
 * No clock, no CRDT: versioning/durability are the persistence port (the product's
 * DB). `toJson`/`applyJson` serialize cells; the version a client tracks is the DB's.
 */

/** The shared slot — the curled `""` (an unset instance, or the option-shared weight). */
export const SHARED = "";

/** A read cursor: one instance per context tier; an absent tier means `""`. */
export type Cursor = Record<string, string>;

/** A cell value: a routable scalar (`f64`) or an opaque document blob (`bytes`). */
export type Value = { f64: number } | { bytes: Uint8Array };

/** Why a write was refused. `"bad-value-kind"` = a seed's declared kind disagrees with
 *  its value payload type (distinct from an unknown tier). */
export type WriteError = "write-up" | "unknown-level" | "bad-value-kind";

function asF64(v: Value): number | undefined {
  return "f64" in v ? v.f64 : undefined;
}
function asBytes(v: Value): Uint8Array | undefined {
  return "bytes" in v ? v.bytes : undefined;
}

/** A fact at a coordinate: the two co-located, sparse layers. */
interface Cell {
  value?: Value;
  weight?: Value;
}

function fcmp(a: number, b: number): number {
  if (Number.isNaN(a) && Number.isNaN(b)) return 0;
  if (Number.isNaN(a)) return 1;
  if (Number.isNaN(b)) return -1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Bytes → standard base64 (RFC 4648, byte-identical to Rust's `base64` crate). */
function bytesToB64(b: Uint8Array): string {
  let s = "";
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]!);
  return btoa(s);
}
function b64ToBytes(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** A wire-encoded value: a number, a `±∞`/`NaN` sentinel, or a `{b64}` document. */
type WireVal = number | string | { b64: string };

function encodeVal(v: Value): WireVal {
  if ("bytes" in v) return { b64: bytesToB64(v.bytes) };
  const x = v.f64;
  if (Number.isFinite(x)) return x;
  if (Number.isNaN(x)) return "nan";
  return x > 0 ? "inf" : "-inf";
}
function decodeVal(v: unknown): Value {
  if (typeof v === "number") return { f64: v };
  if (typeof v === "string") {
    if (v === "inf") return { f64: Infinity };
    if (v === "-inf") return { f64: -Infinity };
    if (v === "nan") return { f64: NaN };
    throw new Error(`bad value sentinel: ${v}`);
  }
  if (v !== null && typeof v === "object" && "b64" in v && typeof (v as { b64: unknown }).b64 === "string") {
    return { bytes: b64ToBytes((v as { b64: string }).b64) };
  }
  throw new Error("bad wire value");
}

/** One serialized cell — what the persistence port stores and ships (no clock). */
export interface JsonCell {
  level: string;
  inst: string;
  option: string;
  dim: string;
  v?: Value;
  w?: Value;
}

/** Compare two strings by Unicode CODE POINT (not UTF-16 code unit). This matches
 *  Rust's `String::cmp` (UTF-8 byte order) for the full Unicode range — including the
 *  astral plane (≥ U+10000), where the naïve `<`/`>` UTF-16 comparator disagrees with
 *  Rust (a surrogate-pair lead unit U+D800..U+DBFF sorts before a BMP private-use char
 *  like U+E000 by code unit, but its code point ≥ U+10000 sorts AFTER by Rust's byte
 *  order). Iterating via `codePointAt` and comparing code points is identical to UTF-8
 *  byte order because UTF-8 preserves code-point order. Load-bearing for the
 *  decision-equivalence LAW: every tie-break / dim sort / pair sort routes through here. */
function scmp(a: string, b: string): number {
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const ca = a.codePointAt(i)!;
    const cb = b.codePointAt(j)!;
    if (ca !== cb) return ca < cb ? -1 : 1;
    // Advance past the full code point (2 UTF-16 units for an astral char).
    i += ca > 0xffff ? 2 : 1;
    j += cb > 0xffff ? 2 : 1;
  }
  // Equal up to the shorter string: the one with characters left sorts AFTER (a prefix
  // sorts before its extension) — same as Rust's byte cmp on a shared prefix.
  if (i < a.length) return 1;
  if (j < b.length) return -1;
  return 0;
}
function cmpCell(a: JsonCell, b: JsonCell): number {
  return scmp(a.level, b.level) || scmp(a.inst, b.inst) || scmp(a.option, b.option) || scmp(a.dim, b.dim);
}

type Cells = Map<string, Map<string, Map<string, Map<string, Cell>>>>;
type Layer = "value" | "weight";

export class Tensor {
  readonly levels: string[];
  private cells: Cells = new Map();

  constructor(levels: string[]) {
    this.levels = [...levels];
  }

  private levelIndex(level: string): number {
    return this.levels.indexOf(level);
  }

  /** Mint a {@link Writer} that may set cells at `level` or finer (write-down). */
  writer(level: string): Writer | undefined {
    const floor = this.levelIndex(level);
    return floor < 0 ? undefined : new Writer(this, floor);
  }

  /** @internal */ indexOf(level: string): number {
    return this.levelIndex(level);
  }
  /** @internal */ put(level: string, inst: string, option: string, dim: string, layer: Layer, value: Value): void {
    let a = this.cells.get(level);
    if (!a) this.cells.set(level, (a = new Map()));
    let b = a.get(inst);
    if (!b) a.set(inst, (b = new Map()));
    let c = b.get(option);
    if (!c) b.set(option, (c = new Map()));
    let cell = c.get(dim);
    if (!cell) c.set(dim, (cell = {}));
    cell[layer] = value;
  }

  private inst(cursor: Cursor, level: string): string {
    return cursor[level] ?? SHARED;
  }
  private get(level: string, inst: string, option: string, dim: string): Cell | undefined {
    return this.cells.get(level)?.get(inst)?.get(option)?.get(dim);
  }

  /** Fold one layer over the context tiers at `cursor`: the deepest set wins. */
  private fold(cursor: Cursor, option: string, dim: string, layer: Layer): Value | undefined {
    let out: Value | undefined;
    for (const level of this.levels) {
      const v = this.get(level, this.inst(cursor, level), option, dim)?.[layer];
      if (v !== undefined) out = v;
    }
    return out;
  }

  /** Effective scalar value at `cursor` (deepest set; a document reads undefined). */
  value(cursor: Cursor, option: string, dim: string): number | undefined {
    const v = this.fold(cursor, option, dim, "value");
    return v === undefined ? undefined : asF64(v);
  }
  /** Effective document bytes at `cursor` (deepest set, if a Bytes cell). */
  bytes(cursor: Cursor, option: string, dim: string): Uint8Array | undefined {
    const v = this.fold(cursor, option, dim, "value");
    return v === undefined ? undefined : asBytes(v);
  }
  /** Effective weight of `(option, dim)`: option-specific over the shared `""` slot. */
  weight(cursor: Cursor, option: string, dim: string): number | undefined {
    const v = this.fold(cursor, option, dim, "weight") ?? this.fold(cursor, SHARED, dim, "weight");
    return v === undefined ? undefined : asF64(v);
  }

  /** Remove the entire `(level, inst)` slice — every option/dim cell at that
   *  coordinate. The orchestrator uses this to prune per-job gate cells from the
   *  long-lived control-plane tensor (a re-routed or completed job must not leave its
   *  `priv:*` cells behind). A no-op if the slice is absent. Mirrors the Rust
   *  `Tensor::clear_level_inst`. */
  clearLevelInst(level: string, inst: string): void {
    this.cells.get(level)?.delete(inst);
  }

  /** The options visible at `cursor` (the shared slot is not an option). */
  options(cursor: Cursor): Set<string> {
    const out = new Set<string>();
    for (const level of this.levels) {
      const m = this.cells.get(level)?.get(this.inst(cursor, level));
      if (m) for (const o of m.keys()) if (o !== SHARED) out.add(o);
    }
    return out;
  }
  /**
   * The dims set for `(cursor, option)`, returned in **sorted** order.
   *
   * The sort is load-bearing for cross-core decision-equivalence, not cosmetic.
   * `resolve` sums `Σ value·weight` over these dims, and IEEE-754 addition is
   * non-associative: a 3+-finite-dim option summed in a different order yields a
   * last-ULP-different score, which can flip a tie and produce the OPPOSITE queue
   * on one core. The Rust core iterates dims via a `BTreeSet` (sorted by name), so
   * TS must add in that same sorted order to stay byte-identical. `scmp` compares by
   * Unicode code point, which is byte-identical to Rust's `String` ordering (UTF-8
   * byte order) over the FULL Unicode range, astral plane included — the same
   * comparator `pairsAt` already uses.
   */
  private dimsOf(cursor: Cursor, option: string): string[] {
    const out = new Set<string>();
    for (const level of this.levels) {
      const m = this.cells.get(level)?.get(this.inst(cursor, level))?.get(option);
      if (m) for (const k of m.keys()) out.add(k);
    }
    return [...out].sort(scmp);
  }

  /** Collapse the dimension axis: Σ value·weight, drop a non-finite gate (`±∞`/`NaN`), order → queue. */
  resolve(cursor: Cursor): string[] {
    const scored: [string, number][] = [];
    for (const option of this.options(cursor)) {
      let score = 0;
      let gated = false;
      let routable = false;
      for (const dim of this.dimsOf(cursor, option)) {
        const v = this.value(cursor, option, dim);
        if (v === undefined) continue;
        routable = true;
        if (!Number.isFinite(v)) {
          gated = true;
          break;
        }
        const w = this.weight(cursor, option, dim);
        if (w !== undefined && Number.isFinite(w)) score += w * v;
      }
      if (!gated && routable) scored.push([option, score]);
    }
    scored.sort((a, b) => fcmp(a[1], b[1]) || scmp(a[0], b[0]));
    return scored.map(([o]) => o);
  }

  /** The one value-path up: median of the value layer over `fromLevel`'s instances. */
  reduceMedian(fromLevel: string, toLevel: string, toInst: string, option: string, dim: string): void {
    const vals: number[] = [];
    const m = this.cells.get(fromLevel);
    if (m) for (const byOpt of m.values()) {
      const cv = byOpt.get(option)?.get(dim)?.value;
      const x = cv === undefined ? undefined : asF64(cv);
      if (x !== undefined && Number.isFinite(x)) vals.push(x);
    }
    if (vals.length === 0) return;
    vals.sort(fcmp);
    const n = vals.length;
    const med = n % 2 === 1 ? vals[n >> 1]! : (vals[(n >> 1) - 1]! + vals[n >> 1]!) / 2;
    this.put(toLevel, toInst, option, dim, "value", { f64: med });
  }

  /** The additive value-path up: the *sum* of the value layer over `fromLevel`'s
   *  instances for `(option, dim)`. The budget layer's feeder — total draw on a shared
   *  resource pool is a sum, not a median. Non-finite values excluded; no support → no write.
   *
   *  The sum order is load-bearing for cross-core decision-equivalence, not cosmetic.
   *  IEEE-754 addition is non-associative, so summing the same draws in a different order
   *  yields a last-ULP-different total — which, fed through `tick → λ → resolve`, can flip
   *  a tie and produce the OPPOSITE queue on one core. Rust iterates instances via a
   *  `BTreeMap` (sorted by name) while TS iterates a `Map` (insertion order), so the
   *  *collection* order already differs across cores (e.g. once ≥10 jobs exist the string
   *  sort `j0,j1,j10,j2,…` ≠ insertion `j0,j1,…,j10`). We therefore **sort the collected
   *  values ascending** before summing: this removes all dependence on map iteration order
   *  and is trivially identical on both cores. Same discipline as `dimsOf` (the resolve sum
   *  order) and `aggregateUsage` (the pool sum order). */
  reduceSum(fromLevel: string, toLevel: string, toInst: string, option: string, dim: string): number | undefined {
    const vals: number[] = [];
    const m = this.cells.get(fromLevel);
    if (m) for (const byOpt of m.values()) {
      const cv = byOpt.get(option)?.get(dim)?.value;
      const x = cv === undefined ? undefined : asF64(cv);
      if (x !== undefined && Number.isFinite(x)) vals.push(x);
    }
    if (vals.length === 0) return undefined;
    // Canonical (ascending) order so both cores add identical operands in identical
    // order — IEEE-754 addition is non-associative; see the doc comment above.
    vals.sort(fcmp);
    let sum = 0;
    for (const x of vals) sum += x;
    this.put(toLevel, toInst, option, dim, "value", { f64: sum });
    return sum;
  }

  /** reduceMedian for every scalar `(option, dim)` at `fromLevel`. */
  rollUp(fromLevel: string, toLevel: string, toInst: string): void {
    for (const [opt, dim] of this.pairsAt(fromLevel)) this.reduceMedian(fromLevel, toLevel, toInst, opt, dim);
  }

  /** The option-axis sibling of reduce: rolls *existence* up. For each subject
   *  `(option, dim)` attested at `fromLevel`, count independent reporters
   *  (instance ≠ option — self-excluded) whose value attests (> 0); if the count
   *  reaches `quorum`, materialize the subject as a corroborated `1.0`. */
  corroborate(fromLevel: string, toLevel: string, toInst: string, quorum: number): void {
    for (const [opt, dim] of this.pairsAt(fromLevel)) {
      let count = 0;
      const m = this.cells.get(fromLevel);
      if (m) for (const [inst, byOpt] of m) {
        if (inst === opt) continue; // self-exclusion
        const cv = byOpt.get(opt)?.get(dim)?.value;
        const x = cv === undefined ? undefined : asF64(cv);
        if (x !== undefined && x > 0) count++;
      }
      if (count >= quorum) this.put(toLevel, toInst, opt, dim, "value", { f64: 1.0 });
    }
  }

  /** The distinct `(option, dim)` pairs at `fromLevel`, in canonical order. */
  private pairsAt(fromLevel: string): [string, string][] {
    const seen = new Set<string>();
    const pairs: [string, string][] = [];
    const m = this.cells.get(fromLevel);
    if (m) for (const byOpt of m.values()) {
      for (const [opt, byDim] of byOpt) {
        for (const dim of byDim.keys()) {
          const k = `${opt} ${dim}`;
          if (!seen.has(k)) {
            seen.add(k);
            pairs.push([opt, dim]);
          }
        }
      }
    }
    pairs.sort((a, b) => scmp(a[0], b[0]) || scmp(a[1], b[1]));
    return pairs;
  }

  /** How many instances at `fromLevel` carry a finite value for `(option, dim)`. */
  support(fromLevel: string, option: string, dim: string): number {
    let n = 0;
    const m = this.cells.get(fromLevel);
    if (m) for (const byOpt of m.values()) {
      const cv = byOpt.get(option)?.get(dim)?.value;
      const x = cv === undefined ? undefined : asF64(cv);
      if (x !== undefined && Number.isFinite(x)) n++;
    }
    return n;
  }

  // ---- Serialization (the persistence port's currency; no clocks) ----------

  /** Every cell as a flat, canonically-sorted list — what the port persists/ships. */
  cells_(): JsonCell[] {
    const out: JsonCell[] = [];
    for (const [lv, byInst] of this.cells)
      for (const [inst, byOpt] of byInst)
        for (const [opt, byDim] of byOpt)
          for (const [dim, cell] of byDim)
            out.push({ level: lv, inst, option: opt, dim, v: cell.value, w: cell.weight });
    out.sort(cmpCell);
    return out;
  }

  /** Serialize the whole tensor (`{levels, cells}`) for storage or a client ship. */
  toJson(): string {
    const cells = this.cells_().map((c) => {
      const o: Record<string, unknown> = { level: c.level, inst: c.inst, option: c.option, dim: c.dim };
      if (c.v !== undefined) o.v = encodeVal(c.v);
      if (c.w !== undefined) o.w = encodeVal(c.w);
      return o;
    });
    return `{"levels":${JSON.stringify(this.levels)},"cells":${JSON.stringify(cells)}}`;
  }

  /** Overlay a list of cells (a snapshot or DB delta) onto this tensor — a plain
   *  overwrite (the DB already decided which row is current). Malformed cells skipped. */
  applyJson(cellsJson: string): void {
    let arr: unknown;
    try {
      arr = JSON.parse(cellsJson);
    } catch {
      return;
    }
    if (!Array.isArray(arr)) return;
    for (const raw of arr) {
      if (typeof raw !== "object" || raw === null) continue;
      const o = raw as Record<string, unknown>;
      if (typeof o.level !== "string" || typeof o.inst !== "string" || typeof o.option !== "string" || typeof o.dim !== "string") {
        continue;
      }
      try {
        // ALL-OR-NOTHING, matching Rust's atomic serde decode (`from_value` decodes the
        // whole JsonCell or fails wholesale): decode BOTH layers FIRST, then put each
        // only after both decodes succeed. A valid-v / malformed-w cell must drop the
        // ENTIRE cell on both cores, not land v while losing w (a parity divergence).
        const vv = o.v !== undefined ? decodeVal(o.v) : undefined;
        const wv = o.w !== undefined ? decodeVal(o.w) : undefined;
        if (vv !== undefined) this.put(o.level, o.inst, o.option, o.dim, "value", vv);
        if (wv !== undefined) this.put(o.level, o.inst, o.option, o.dim, "weight", wv);
      } catch {
        // malformed cell — skip
      }
    }
  }

  /** Rebuild a tensor from {@link Tensor.toJson}. */
  static fromJson(s: string): Tensor | undefined {
    let v: unknown;
    try {
      v = JSON.parse(s);
    } catch {
      return undefined;
    }
    if (typeof v !== "object" || v === null) return undefined;
    const o = v as Record<string, unknown>;
    if (!Array.isArray(o.levels) || !o.levels.every((l) => typeof l === "string")) return undefined;
    const t = new Tensor(o.levels as string[]);
    t.applyJson(JSON.stringify(o.cells ?? []));
    return t;
  }
}

/** A capability handle: writes at its floor tier or finer, never coarser. No clock —
 *  the DB versions the write; this enforces only write-down. */
export class Writer {
  constructor(
    private tensor: Tensor,
    private floor: number,
  ) {}

  private check(level: string): WriteError | undefined {
    const i = this.tensor.indexOf(level);
    if (i < 0) return "unknown-level";
    if (i < this.floor) return "write-up";
    return undefined;
  }
  private write(level: string, inst: string, option: string, dim: string, layer: Layer, value: Value): WriteError | undefined {
    const e = this.check(level);
    if (e) return e;
    this.tensor.put(level, inst, option, dim, layer, value);
    return undefined;
  }

  setValue(level: string, inst: string, option: string, dim: string, v: number): WriteError | undefined {
    return this.write(level, inst, option, dim, "value", { f64: v });
  }
  setBytes(level: string, inst: string, option: string, dim: string, b: Uint8Array): WriteError | undefined {
    return this.write(level, inst, option, dim, "value", { bytes: b });
  }
  setWeight(level: string, inst: string, dim: string, w: number): WriteError | undefined {
    return this.write(level, inst, SHARED, dim, "weight", { f64: w });
  }
  setOptionWeight(level: string, inst: string, option: string, dim: string, w: number): WriteError | undefined {
    return this.write(level, inst, option, dim, "weight", { f64: w });
  }

  attenuate(level: string): Writer | undefined {
    const i = this.tensor.indexOf(level);
    return i < 0 ? undefined : new Writer(this.tensor, Math.max(i, this.floor));
  }
}
