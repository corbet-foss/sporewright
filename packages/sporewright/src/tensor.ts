// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * The stacked tensor of scoped preference layers.
 *
 * A cell is addressed by `(layer, scope, option, dimension)`. `scope` carries
 * owner coordinates plus free context facets, allowing feedback at a fine layer
 * to remain conditional on the path and circumstances where it was learned.
 */

export const SHARED = "";

export type Scope = Record<string, string>;
export type Cursor = Scope;

/** Build a scope while omitting empty coordinates. */
export function scope(coordinates: Scope = {}): Scope {
  return Object.fromEntries(Object.entries(coordinates).filter(([, value]) => value !== ""));
}

export type Value = { f64: number } | { bytes: Uint8Array };
export type WriteError = "write-up" | "unknown-layer" | "scope-outside-layer" | "bad-value-kind";

export interface JsonCell {
  layer: string;
  scope: Scope;
  option: string;
  dim: string;
  v?: Value;
  w?: Value;
}

export interface Origin {
  layer: string;
  scope: Scope;
}

export interface DimensionTrace {
  dimension: string;
  value: number;
  weight?: number;
  contribution: number;
  value_origin: Origin;
  weight_origin?: Origin;
}

export interface ResolvedOption {
  option: string;
  viable: boolean;
  score: number;
  dimensions: DimensionTrace[];
}

export interface PreferenceChange {
  layer: string;
  scope: Scope;
  option: string;
  dimension: string;
  previous?: number;
  next: number;
}

interface Cell {
  value?: Value;
  weight?: Value;
}

interface CellKey {
  layer: string;
  scope: [string, string][];
  option: string;
  dim: string;
}

interface Entry {
  key: CellKey;
  cell: Cell;
}

type Plane = "value" | "weight";
type WireVal = number | string | { b64: string };

function asF64(value: Value): number | undefined {
  return "f64" in value ? value.f64 : undefined;
}

function asBytes(value: Value): Uint8Array | undefined {
  return "bytes" in value ? value.bytes : undefined;
}

function fcmp(left: number, right: number): number {
  if (Number.isNaN(left) && Number.isNaN(right)) return 0;
  if (Number.isNaN(left)) return 1;
  if (Number.isNaN(right)) return -1;
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Unicode code-point order, decision-equivalent to Rust UTF-8 string order. */
function scmp(left: string, right: string): number {
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const a = left.codePointAt(i)!;
    const b = right.codePointAt(j)!;
    if (a !== b) return a < b ? -1 : 1;
    i += a > 0xffff ? 2 : 1;
    j += b > 0xffff ? 2 : 1;
  }
  if (i < left.length) return 1;
  if (j < right.length) return -1;
  return 0;
}

function bytesToB64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function b64ToBytes(encoded: string): Uint8Array {
  const binary = atob(encoded);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

function encodeVal(value: Value): WireVal {
  if ("bytes" in value) return { b64: bytesToB64(value.bytes) };
  if (Number.isFinite(value.f64)) return value.f64;
  if (Number.isNaN(value.f64)) return "nan";
  return value.f64 > 0 ? "inf" : "-inf";
}

function decodeVal(value: unknown): Value {
  if (typeof value === "number") return { f64: value };
  if (value === "inf") return { f64: Infinity };
  if (value === "-inf") return { f64: -Infinity };
  if (value === "nan") return { f64: NaN };
  if (
    value !== null
    && typeof value === "object"
    && "b64" in value
    && typeof (value as { b64: unknown }).b64 === "string"
  ) return { bytes: b64ToBytes((value as { b64: string }).b64) };
  throw new Error("bad wire value");
}

function cloneScope(input: Scope): Scope {
  return { ...input };
}

export class Tensor {
  readonly layers: string[];
  private readonly entries = new Map<string, Entry>();

  constructor(layers: string[]) {
    this.layers = [...layers];
  }

  private layerIndex(layer: string): number {
    return this.layers.indexOf(layer);
  }

  /** @internal */ indexOf(layer: string): number {
    return this.layerIndex(layer);
  }

  private normalizedScope(layer: string, input: Scope): [string, string][] | WriteError {
    const owner = this.layerIndex(layer);
    if (owner < 0) return "unknown-layer";
    const normalized: [string, string][] = [];
    for (const [axis, value] of Object.entries(input)) {
      if (value === "") continue;
      const index = this.layerIndex(axis);
      if (index > owner) return "scope-outside-layer";
      normalized.push([axis, value]);
    }
    normalized.sort((left, right) => {
      const leftIndex = this.layerIndex(left[0]);
      const rightIndex = this.layerIndex(right[0]);
      const leftRank = leftIndex < 0 ? Number.MAX_SAFE_INTEGER : leftIndex;
      const rightRank = rightIndex < 0 ? Number.MAX_SAFE_INTEGER : rightIndex;
      return leftRank - rightRank || scmp(left[0], right[0]);
    });
    return normalized;
  }

  private key(layer: string, input: Scope, option: string, dim: string): CellKey | WriteError {
    const normalized = this.normalizedScope(layer, input);
    if (typeof normalized === "string") return normalized;
    return { layer, scope: normalized, option, dim };
  }

  private keyId(key: CellKey): string {
    return JSON.stringify([key.layer, key.scope, key.option, key.dim]);
  }

  /** @internal */ put(
    layer: string,
    input: Scope,
    option: string,
    dim: string,
    plane: Plane,
    value: Value,
  ): WriteError | undefined {
    const key = this.key(layer, input, option, dim);
    if (typeof key === "string") return key;
    const id = this.keyId(key);
    const entry = this.entries.get(id) ?? { key, cell: {} };
    entry.cell[plane] = value;
    this.entries.set(id, entry);
    return undefined;
  }

  writer(floor: string): Writer | undefined {
    const index = this.layerIndex(floor);
    return index < 0 ? undefined : new Writer(this, index);
  }

  clearScope(layer: string, input: Scope): WriteError | undefined {
    const normalized = this.normalizedScope(layer, input);
    if (typeof normalized === "string") return normalized;
    for (const [id, entry] of this.entries) {
      if (entry.key.layer === layer && this.scopeCmp(entry.key.scope, normalized) === 0) {
        this.entries.delete(id);
      }
    }
    return undefined;
  }

  private matches(candidate: [string, string][], cursor: Cursor): boolean {
    return candidate.every(([axis, value]) => cursor[axis] === value);
  }

  private scopeCmp(left: [string, string][], right: [string, string][]): number {
    const length = Math.min(left.length, right.length);
    for (let index = 0; index < length; index++) {
      const axis = scmp(left[index]![0], right[index]![0]);
      if (axis !== 0) return axis;
      const value = scmp(left[index]![1], right[index]![1]);
      if (value !== 0) return value;
    }
    return left.length - right.length;
  }

  private specificityCmp(left: CellKey, right: CellKey): number {
    const layer = this.layerIndex(left.layer) - this.layerIndex(right.layer);
    if (layer !== 0) return layer;
    for (const axis of [...this.layers].reverse()) {
      const leftHas = left.scope.some(([candidate]) => candidate === axis);
      const rightHas = right.scope.some(([candidate]) => candidate === axis);
      if (leftHas !== rightHas) return leftHas ? 1 : -1;
    }
    return left.scope.length - right.scope.length || this.scopeCmp(left.scope, right.scope);
  }

  private best(cursor: Cursor, option: string, dim: string, plane: Plane): Entry | undefined {
    let best: Entry | undefined;
    for (const entry of this.entries.values()) {
      if (
        entry.key.option !== option
        || entry.key.dim !== dim
        || entry.cell[plane] === undefined
        || !this.matches(entry.key.scope, cursor)
      ) continue;
      if (best === undefined || this.specificityCmp(entry.key, best.key) > 0) best = entry;
    }
    return best;
  }

  private origin(key: CellKey): Origin {
    return { layer: key.layer, scope: Object.fromEntries(key.scope) };
  }

  value(cursor: Cursor, option: string, dim: string): number | undefined {
    const value = this.best(cursor, option, dim, "value")?.cell.value;
    return value === undefined ? undefined : asF64(value);
  }

  bytes(cursor: Cursor, option: string, dim: string): Uint8Array | undefined {
    const value = this.best(cursor, option, dim, "value")?.cell.value;
    return value === undefined ? undefined : asBytes(value);
  }

  weight(cursor: Cursor, option: string, dim: string): number | undefined {
    const value = this.best(cursor, option, dim, "weight")?.cell.weight
      ?? this.best(cursor, SHARED, dim, "weight")?.cell.weight;
    return value === undefined ? undefined : asF64(value);
  }

  options(cursor: Cursor): Set<string> {
    const options = new Set<string>();
    for (const { key } of this.entries.values()) {
      if (key.option !== SHARED && this.matches(key.scope, cursor)) options.add(key.option);
    }
    return options;
  }

  resolveExplained(cursor: Cursor): ResolvedOption[] {
    // Fold compatible cells once. Re-running `best` for every option/dimension
    // pair made resolution quadratic in a large sparse participant tensor.
    const dimensionsByOption = new Map<string, Set<string>>();
    const values = new Map<string, Entry>();
    const weights = new Map<string, Entry>();
    const pairId = (option: string, dimension: string) => JSON.stringify([option, dimension]);
    const keepBest = (target: Map<string, Entry>, id: string, candidate: Entry) => {
      const current = target.get(id);
      if (current === undefined || this.specificityCmp(candidate.key, current.key) > 0) {
        target.set(id, candidate);
      }
    };
    for (const entry of this.entries.values()) {
      if (!this.matches(entry.key.scope, cursor)) continue;
      if (entry.key.option !== SHARED) {
        const optionDimensions = dimensionsByOption.get(entry.key.option) ?? new Set<string>();
        optionDimensions.add(entry.key.dim);
        dimensionsByOption.set(entry.key.option, optionDimensions);
      }
      const id = pairId(entry.key.option, entry.key.dim);
      if (entry.cell.value !== undefined) keepBest(values, id, entry);
      if (entry.cell.weight !== undefined) keepBest(weights, id, entry);
    }

    const resolved: ResolvedOption[] = [];
    for (const option of [...dimensionsByOption.keys()].sort(scmp)) {
      let score = 0;
      let viable = true;
      let routable = false;
      const dimensions: DimensionTrace[] = [];
      for (const dimension of [...dimensionsByOption.get(option)!].sort(scmp)) {
        const valueEntry = values.get(pairId(option, dimension));
        const valueCell = valueEntry?.cell.value;
        const value = valueCell === undefined ? undefined : asF64(valueCell);
        if (valueEntry === undefined || value === undefined) continue;
        routable = true;
        const weightEntry = weights.get(pairId(option, dimension))
          ?? weights.get(pairId(SHARED, dimension));
        const weightCell = weightEntry?.cell.weight;
        const weight = weightCell === undefined ? undefined : asF64(weightCell);
        const contribution = weight !== undefined && Number.isFinite(weight) && Number.isFinite(value)
          ? value * weight
          : 0;
        if (!Number.isFinite(value)) viable = false;
        score += contribution;
        dimensions.push({
          dimension,
          value,
          weight,
          contribution,
          value_origin: this.origin(valueEntry.key),
          weight_origin: weightEntry === undefined ? undefined : this.origin(weightEntry.key),
        });
      }
      if (routable) resolved.push({ option, viable, score, dimensions });
    }
    resolved.sort((left, right) => {
      if (left.viable !== right.viable) return left.viable ? -1 : 1;
      return fcmp(left.score, right.score) || scmp(left.option, right.option);
    });
    return resolved;
  }

  resolve(cursor: Cursor): string[] {
    return this.resolveExplained(cursor).filter(({ viable }) => viable).map(({ option }) => option);
  }

  private isDescendant(candidate: [string, string][], ancestor: Scope): boolean {
    return Object.entries(ancestor)
      .filter(([, value]) => value !== "")
      .every(([axis, value]) => candidate.some(([currentAxis, current]) => currentAxis === axis && current === value));
  }

  private checkReduction(fromLayer: string, toLayer: string, targetScope: Scope): WriteError | undefined {
    const from = this.layerIndex(fromLayer);
    const to = this.layerIndex(toLayer);
    if (from < 0 || to < 0) return "unknown-layer";
    if (from <= to) return "write-up";
    const normalized = this.normalizedScope(toLayer, targetScope);
    return typeof normalized === "string" ? normalized : undefined;
  }

  private descendantValues(fromLayer: string, ancestor: Scope, option: string, dim: string): number[] {
    const values: number[] = [];
    for (const { key, cell } of this.entries.values()) {
      if (
        key.layer !== fromLayer
        || key.option !== option
        || key.dim !== dim
        || !this.isDescendant(key.scope, ancestor)
        || cell.value === undefined
      ) continue;
      const value = asF64(cell.value);
      if (value !== undefined && Number.isFinite(value)) values.push(value);
    }
    return values;
  }

  /** @internal */ exactValue(layer: string, input: Scope, option: string, dim: string): number | undefined {
    const key = this.key(layer, input, option, dim);
    if (typeof key === "string") return undefined;
    const value = this.entries.get(this.keyId(key))?.cell.value;
    return value === undefined ? undefined : asF64(value);
  }

  private replaceReducedValue(
    layer: string,
    targetScope: Scope,
    option: string,
    dimension: string,
    next: number,
  ): PreferenceChange | WriteError {
    const previous = this.exactValue(layer, targetScope, option, dimension);
    const error = this.put(layer, targetScope, option, dimension, "value", { f64: next });
    if (error) return error;
    return { layer, scope: cloneScope(targetScope), option, dimension, previous, next };
  }

  reduceMedian(
    fromLayer: string,
    toLayer: string,
    targetScope: Scope,
    option: string,
    dim: string,
  ): PreferenceChange | WriteError | undefined {
    const error = this.checkReduction(fromLayer, toLayer, targetScope);
    if (error) return error;
    const values = this.descendantValues(fromLayer, targetScope, option, dim).sort(fcmp);
    if (values.length === 0) return undefined;
    const middle = values.length >> 1;
    const median = values.length % 2 === 1
      ? values[middle]!
      : (values[middle - 1]! + values[middle]!) / 2;
    return this.replaceReducedValue(toLayer, targetScope, option, dim, median);
  }

  reduceSum(
    fromLayer: string,
    toLayer: string,
    targetScope: Scope,
    option: string,
    dim: string,
  ): PreferenceChange | WriteError | undefined {
    const error = this.checkReduction(fromLayer, toLayer, targetScope);
    if (error) return error;
    const values = this.descendantValues(fromLayer, targetScope, option, dim).sort(fcmp);
    if (values.length === 0) return undefined;
    let sum = 0;
    for (const value of values) sum += value;
    return this.replaceReducedValue(toLayer, targetScope, option, dim, sum);
  }

  private descendantPairs(fromLayer: string, ancestor: Scope): [string, string][] {
    const pairs = new Map<string, [string, string]>();
    for (const { key } of this.entries.values()) {
      if (key.layer === fromLayer && key.option !== SHARED && this.isDescendant(key.scope, ancestor)) {
        pairs.set(JSON.stringify([key.option, key.dim]), [key.option, key.dim]);
      }
    }
    return [...pairs.values()].sort((left, right) => scmp(left[0], right[0]) || scmp(left[1], right[1]));
  }

  reduceAllMedian(fromLayer: string, toLayer: string, targetScope: Scope): PreferenceChange[] | WriteError {
    const error = this.checkReduction(fromLayer, toLayer, targetScope);
    if (error) return error;
    const changes: PreferenceChange[] = [];
    for (const [option, dim] of this.descendantPairs(fromLayer, targetScope)) {
      const change = this.reduceMedian(fromLayer, toLayer, targetScope, option, dim);
      if (typeof change === "string") return change;
      if (change !== undefined) changes.push(change);
    }
    return changes;
  }

  support(fromLayer: string, ancestor: Scope, option: string, dim: string): number {
    return this.descendantValues(fromLayer, ancestor, option, dim).length;
  }

  corroborate(
    fromLayer: string,
    toLayer: string,
    targetScope: Scope,
    option: string,
    dim: string,
    reporterAxis: string,
    quorum: number,
  ): PreferenceChange | WriteError | undefined {
    const error = this.checkReduction(fromLayer, toLayer, targetScope);
    if (error) return error;
    const reporters = new Set<string>();
    for (const { key, cell } of this.entries.values()) {
      if (
        key.layer !== fromLayer
        || key.option !== option
        || key.dim !== dim
        || !this.isDescendant(key.scope, targetScope)
        || cell.value === undefined
        || (asF64(cell.value) ?? 0) <= 0
      ) continue;
      const reporter = key.scope.find(([axis]) => axis === reporterAxis)?.[1];
      if (reporter !== undefined && reporter !== option) reporters.add(reporter);
    }
    if (reporters.size < quorum) return undefined;
    return this.replaceReducedValue(toLayer, targetScope, option, dim, 1);
  }

  corroborateAll(
    fromLayer: string,
    toLayer: string,
    targetScope: Scope,
    reporterAxis: string,
    quorum: number,
  ): PreferenceChange[] | WriteError {
    const error = this.checkReduction(fromLayer, toLayer, targetScope);
    if (error) return error;
    const changes: PreferenceChange[] = [];
    for (const [option, dim] of this.descendantPairs(fromLayer, targetScope)) {
      const change = this.corroborate(fromLayer, toLayer, targetScope, option, dim, reporterAxis, quorum);
      if (typeof change === "string") return change;
      if (change !== undefined) changes.push(change);
    }
    return changes;
  }

  cells_(): JsonCell[] {
    const cells = [...this.entries.values()].map(({ key, cell }) => ({
      layer: key.layer,
      scope: Object.fromEntries(key.scope),
      option: key.option,
      dim: key.dim,
      v: cell.value,
      w: cell.weight,
    }));
    cells.sort((left, right) => {
      const layer = scmp(left.layer, right.layer);
      if (layer !== 0) return layer;
      const leftScope = Object.entries(left.scope) as [string, string][];
      const rightScope = Object.entries(right.scope) as [string, string][];
      const scoped = this.scopeCmp(leftScope, rightScope);
      return scoped || scmp(left.option, right.option) || scmp(left.dim, right.dim);
    });
    return cells;
  }

  toJson(): string {
    const cells = this.cells_().map((cell) => {
      const encoded: Record<string, unknown> = {
        layer: cell.layer,
        scope: cell.scope,
        option: cell.option,
        dim: cell.dim,
      };
      if (cell.v !== undefined) encoded.v = encodeVal(cell.v);
      if (cell.w !== undefined) encoded.w = encodeVal(cell.w);
      return encoded;
    });
    return JSON.stringify({ layers: this.layers, cells });
  }

  applyJson(cellsJson: string): void {
    let cells: unknown;
    try {
      cells = JSON.parse(cellsJson);
    } catch {
      return;
    }
    if (!Array.isArray(cells)) return;
    for (const raw of cells) {
      if (raw === null || typeof raw !== "object") continue;
      const cell = raw as Record<string, unknown>;
      if (
        typeof cell.layer !== "string"
        || cell.scope === null
        || typeof cell.scope !== "object"
        || Array.isArray(cell.scope)
        || !Object.values(cell.scope).every((value) => typeof value === "string")
        || typeof cell.option !== "string"
        || typeof cell.dim !== "string"
      ) continue;
      try {
        const value = cell.v === undefined ? undefined : decodeVal(cell.v);
        const weight = cell.w === undefined ? undefined : decodeVal(cell.w);
        const input = cell.scope as Scope;
        if (value !== undefined) this.put(cell.layer, input, cell.option, cell.dim, "value", value);
        if (weight !== undefined) this.put(cell.layer, input, cell.option, cell.dim, "weight", weight);
      } catch {
        // A malformed cell is ignored as a unit.
      }
    }
  }

  static fromJson(snapshot: string): Tensor | undefined {
    let value: unknown;
    try {
      value = JSON.parse(snapshot);
    } catch {
      return undefined;
    }
    if (value === null || typeof value !== "object") return undefined;
    const object = value as Record<string, unknown>;
    if (!Array.isArray(object.layers) || !object.layers.every((layer) => typeof layer === "string")) return undefined;
    const tensor = new Tensor(object.layers as string[]);
    tensor.applyJson(JSON.stringify(object.cells ?? []));
    return tensor;
  }
}

export class Writer {
  constructor(
    private readonly tensor: Tensor,
    private readonly floor: number,
  ) {}

  private write(
    layer: string,
    input: Scope,
    option: string,
    dim: string,
    plane: Plane,
    value: Value,
  ): WriteError | undefined {
    const index = this.tensor.indexOf(layer);
    if (index < 0) return "unknown-layer";
    if (index < this.floor) return "write-up";
    return this.tensor.put(layer, input, option, dim, plane, value);
  }

  setValue(layer: string, input: Scope, option: string, dim: string, value: number): WriteError | undefined {
    return this.write(layer, input, option, dim, "value", { f64: value });
  }

  setBytes(layer: string, input: Scope, option: string, dim: string, value: Uint8Array): WriteError | undefined {
    return this.write(layer, input, option, dim, "value", { bytes: value });
  }

  setWeight(layer: string, input: Scope, dim: string, weight: number): WriteError | undefined {
    return this.write(layer, input, SHARED, dim, "weight", { f64: weight });
  }

  setOptionWeight(
    layer: string,
    input: Scope,
    option: string,
    dim: string,
    weight: number,
  ): WriteError | undefined {
    return this.write(layer, input, option, dim, "weight", { f64: weight });
  }

  nudgeValue(
    layer: string,
    input: Scope,
    option: string,
    dimension: string,
    observation: number,
    rate: number,
  ): PreferenceChange | WriteError {
    const index = this.tensor.indexOf(layer);
    if (index < 0) return "unknown-layer";
    if (index < this.floor) return "write-up";
    const previous = this.tensor.exactValue(layer, input, option, dimension);
    const bounded = Number.isFinite(rate) ? Math.min(Math.max(rate, 0), 1) : 1;
    const next = previous !== undefined && Number.isFinite(previous) && Number.isFinite(observation)
      ? previous + bounded * (observation - previous)
      : observation;
    const error = this.tensor.put(layer, input, option, dimension, "value", { f64: next });
    if (error) return error;
    return { layer, scope: cloneScope(input), option, dimension, previous, next };
  }

  attenuate(layer: string): Writer | undefined {
    const index = this.tensor.indexOf(layer);
    return index < 0 ? undefined : new Writer(this.tensor, Math.max(index, this.floor));
  }
}
