// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/** Sparse addressed residual fields with cached Gaussian message passing. */

export const FIELD_SHARED = "";

export type Address = Record<string, string>;

export function address(coordinates: Address = {}): Address {
  return Object.fromEntries(Object.entries(coordinates).filter(([, value]) => value !== ""));
}

export type FieldError =
  | "unknown-layer"
  | "unknown-coordinate"
  | "address-gap"
  | "address-too-shallow"
  | "write-up"
  | "invalid-variance"
  | "invalid-observation"
  | "invalid-discount"
  | "no-evidence"
  | "root-compaction"
  | "frozen-process-variance"
  | "invalid-schema";

export interface Natural {
  precision: number;
  information: number;
}

export interface Evidence extends Natural {
  observations: number;
}

export interface JsonFieldCell {
  address: Address;
  option: string;
  dimension: string;
  prior?: number;
  weight?: number;
  gate?: boolean;
  evidence: Evidence;
}

export interface ResidualTrace {
  address: Address;
  posterior_mean: number;
  posterior_variance: number;
  residual_mean: number;
  declared_prior: number;
  observations: number;
}

export interface FieldDimensionTrace {
  dimension: string;
  mean: number;
  variance: number;
  weight: number;
  contribution: number;
  contributions: ResidualTrace[];
}

export interface FieldResolvedOption {
  option: string;
  viable: boolean;
  expected_score: number;
  score_variance: number;
  decision_score: number;
  dimensions: FieldDimensionTrace[];
}

export interface DecisionPolicy {
  temperature: number;
}

export interface ResolveWork {
  address_nodes: number;
  stored_cells_visited: number;
  parameter_lookups: number;
  ranked_options: number;
}

export interface FieldDecision {
  address: Address;
  policy: DecisionPolicy;
  alternatives: FieldResolvedOption[];
  work: ResolveWork;
}

export interface RoutingReceipt {
  tensor_revision: string;
  decision: FieldDecision;
  selected: string[];
}

export interface FeedbackTrace {
  address: Address;
  option: string;
  dimension: string;
  observed: number;
  observation_variance: number;
  updated_nodes: number;
  posterior_before?: [number, number];
  posterior_after: [number, number];
}

export interface DiscountTrace {
  address: Address;
  option: string;
  dimension: string;
  factor: number;
  precision_before: number;
  precision_after: number;
  updated_nodes: number;
}

export interface CompactionTrace {
  address: Address;
  compacted_into: Address;
  option: string;
  dimension: string;
  observations: number;
  removed_cells: number;
  updated_nodes: number;
}

interface Gaussian {
  mean: number;
  variance: number;
}

interface Cell {
  prior?: number;
  weight?: number;
  gate?: boolean;
  evidence: Evidence;
  childMessages: Natural;
  messageToParent: Natural;
}

interface CellKey {
  path: string[];
  option: string;
  dimension: string;
}

interface Entry {
  key: CellKey;
  cell: Cell;
}

const ZERO_NATURAL = (): Natural => ({ precision: 0, information: 0 });
const ZERO_EVIDENCE = (): Evidence => ({ precision: 0, information: 0, observations: 0 });

function addNatural(left: Natural, right: Natural): Natural {
  return {
    precision: left.precision + right.precision,
    information: left.information + right.information,
  };
}

function subNatural(left: Natural, right: Natural): Natural {
  return {
    precision: Math.max(0, left.precision - right.precision),
    information: left.information - right.information,
  };
}

function naturalFromGaussian(gaussian: Gaussian): Natural {
  const precision = 1 / gaussian.variance;
  return { precision, information: gaussian.mean * precision };
}

function gaussianFromNatural(natural: Natural): Gaussian | undefined {
  if (!(natural.precision > 0) || !Number.isFinite(natural.precision)) return undefined;
  return {
    mean: natural.information / natural.precision,
    variance: 1 / natural.precision,
  };
}

function combine(prior: Gaussian, likelihood: Natural): Gaussian {
  return gaussianFromNatural(addNatural(naturalFromGaussian(prior), likelihood)) ?? prior;
}

function compareStrings(left: string, right: string): number {
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const a = left.codePointAt(i)!;
    const b = right.codePointAt(j)!;
    if (a !== b) return a < b ? -1 : 1;
    i += a > 0xffff ? 2 : 1;
    j += b > 0xffff ? 2 : 1;
  }
  return i < left.length ? 1 : j < right.length ? -1 : 0;
}

function newCell(): Cell {
  return {
    evidence: ZERO_EVIDENCE(),
    childMessages: ZERO_NATURAL(),
    messageToParent: ZERO_NATURAL(),
  };
}

function isFieldError(value: unknown): value is FieldError {
  return typeof value === "string";
}

/**
 * One sparse field. Reads visit only exact prefix buckets on the requested
 * address; observations update cached messages on one leaf-to-root path.
 */
export class AddressedField {
  readonly layers: string[];
  private defaultRootProcessVariance = 1;
  private defaultProcessVariance: number[];
  private readonly dimensionRootProcessVariance = new Map<string, number>();
  private readonly dimensionProcessVariance = new Map<string, number[]>();
  private readonly cells = new Map<string, Entry>();
  private readonly stateDimensions = new Map<string, Set<string>>();
  private readonly pathIndex = new Map<string, Map<string, [string, string]>>();
  private processVarianceFrozen = false;

  constructor(layers: string[]) {
    if (layers.some((layer) => layer.length === 0)
      || new Set(layers).size !== layers.length) throw new Error("invalid-schema");
    this.layers = [...layers];
    this.defaultProcessVariance = layers.map(() => 1);
  }

  private layerIndex(layer: string): number {
    return this.layers.indexOf(layer);
  }

  /** @internal */ indexOf(layer: string): number {
    return this.layerIndex(layer);
  }

  /** @internal */ path(input: Address): string[] | FieldError {
    if (Object.keys(input).some((axis) => this.layerIndex(axis) < 0)) return "unknown-coordinate";
    const result: string[] = [];
    let gap = false;
    for (const layer of this.layers) {
      const value = input[layer];
      if (value !== undefined && value !== "") {
        if (gap) return "address-gap";
        result.push(value);
      } else {
        gap = true;
      }
    }
    return result;
  }

  private addressForPath(path: string[]): Address {
    return Object.fromEntries(path.map((value, index) => [this.layers[index]!, value]));
  }

  private key(path: string[], option: string, dimension: string): CellKey {
    return { path: [...path], option, dimension };
  }

  private keyId(path: string[], option: string, dimension: string): string {
    return JSON.stringify([path, option, dimension]);
  }

  private pathId(path: string[]): string {
    return JSON.stringify(path);
  }

  private stateDimensionId(path: string[], option: string): string {
    return JSON.stringify([path, option]);
  }

  private entry(path: string[], option: string, dimension: string): Entry | undefined {
    return this.cells.get(this.keyId(path, option, dimension));
  }

  private processVariance(dimension: string, depth: number): number {
    return (this.dimensionProcessVariance.get(dimension) ?? this.defaultProcessVariance)[depth]!;
  }

  private rootProcessVariance(dimension: string): number {
    return this.dimensionRootProcessVariance.get(dimension) ?? this.defaultRootProcessVariance;
  }

  setDefaultRootProcessVariance(variance: number): FieldError | undefined {
    if (this.processVarianceFrozen) return "frozen-process-variance";
    if (!Number.isFinite(variance) || variance <= 0) return "invalid-variance";
    this.defaultRootProcessVariance = variance;
    this.rebuildMessages();
    return undefined;
  }

  setDimensionRootProcessVariance(dimension: string, variance: number): FieldError | undefined {
    if (this.processVarianceFrozen) return "frozen-process-variance";
    if (!Number.isFinite(variance) || variance <= 0) return "invalid-variance";
    this.dimensionRootProcessVariance.set(dimension, variance);
    this.rebuildMessages();
    return undefined;
  }

  setDefaultProcessVariance(layer: string, variance: number): FieldError | undefined {
    if (this.processVarianceFrozen) return "frozen-process-variance";
    if (!Number.isFinite(variance) || variance <= 0) return "invalid-variance";
    const index = this.layerIndex(layer);
    if (index < 0) return "unknown-layer";
    this.defaultProcessVariance[index] = variance;
    this.rebuildMessages();
    return undefined;
  }

  setDimensionProcessVariance(dimension: string, layer: string, variance: number): FieldError | undefined {
    if (this.processVarianceFrozen) return "frozen-process-variance";
    if (!Number.isFinite(variance) || variance <= 0) return "invalid-variance";
    const index = this.layerIndex(layer);
    if (index < 0) return "unknown-layer";
    const values = this.dimensionProcessVariance.get(dimension) ?? [...this.defaultProcessVariance];
    values[index] = variance;
    this.dimensionProcessVariance.set(dimension, values);
    this.rebuildMessages();
    return undefined;
  }

  writer(floor: string): FieldWriter | undefined {
    const index = this.layerIndex(floor);
    return index < 0 ? undefined : new FieldWriter(this, index);
  }

  /** A product-authority writer that can declare policy at the implicit root. */
  rootWriter(): FieldWriter {
    return new FieldWriter(this, undefined);
  }

  private ensurePath(path: string[], option: string, dimension: string): void {
    for (let depth = 0; depth <= path.length; depth++) {
      const prefix = path.slice(0, depth);
      const id = this.keyId(prefix, option, dimension);
      if (!this.cells.has(id)) this.cells.set(id, { key: this.key(prefix, option, dimension), cell: newCell() });
      const dimensionId = this.stateDimensionId(prefix, option);
      const dimensions = this.stateDimensions.get(dimensionId) ?? new Set<string>();
      dimensions.add(dimension);
      this.stateDimensions.set(dimensionId, dimensions);
    }
  }

  private indexCoordinate(path: string[], option: string, dimension: string): void {
    const id = this.pathId(path);
    const bucket = this.pathIndex.get(id) ?? new Map<string, [string, string]>();
    bucket.set(JSON.stringify([option, dimension]), [option, dimension]);
    this.pathIndex.set(id, bucket);
  }

  private subtree(cell: Cell): Natural {
    return addNatural(cell.evidence, cell.childMessages);
  }

  private messageFor(key: CellKey): Natural {
    if (key.path.length === 0) return ZERO_NATURAL();
    const entry = this.entry(key.path, key.option, key.dimension);
    if (entry === undefined) return ZERO_NATURAL();
    const like = gaussianFromNatural(this.subtree(entry.cell));
    if (like === undefined) return ZERO_NATURAL();
    const depth = key.path.length - 1;
    return naturalFromGaussian({
      mean: like.mean - (entry.cell.prior ?? 0),
      variance: like.variance + this.processVariance(key.dimension, depth),
    });
  }

  private recomputeUp(path: string[], option: string, dimension: string): number {
    let updated = 0;
    for (let depth = path.length; depth >= 0; depth--) {
      const prefix = path.slice(0, depth);
      const entry = this.entry(prefix, option, dimension)!;
      const previous = entry.cell.messageToParent;
      const next = this.messageFor(entry.key);
      entry.cell.messageToParent = next;
      updated++;
      if (depth > 0) {
        const parent = this.entry(path.slice(0, depth - 1), option, dimension)!;
        parent.cell.childMessages = addNatural(subNatural(parent.cell.childMessages, previous), next);
      }
    }
    return updated;
  }

  private rebuildMessages(): void {
    for (const { cell } of this.cells.values()) {
      cell.childMessages = ZERO_NATURAL();
      cell.messageToParent = ZERO_NATURAL();
    }
    const entries = [...this.cells.values()].sort((left, right) =>
      right.key.path.length - left.key.path.length
      || compareStrings(this.keyId(left.key.path, left.key.option, left.key.dimension), this.keyId(right.key.path, right.key.option, right.key.dimension))
    );
    for (const entry of entries) {
      const next = this.messageFor(entry.key);
      entry.cell.messageToParent = next;
      if (entry.key.path.length > 0) {
        const parent = this.entry(entry.key.path.slice(0, -1), entry.key.option, entry.key.dimension)!;
        parent.cell.childMessages = addNatural(parent.cell.childMessages, next);
      }
    }
  }

  /** @internal */ setPrior(path: string[], option: string, dimension: string, value: number): FieldError | undefined {
    if (!Number.isFinite(value)) return "invalid-observation";
    this.ensurePath(path, option, dimension);
    this.indexCoordinate(path, option, dimension);
    this.entry(path, option, dimension)!.cell.prior = value;
    this.recomputeUp(path, option, dimension);
    return undefined;
  }

  /** @internal */ setWeight(path: string[], option: string, dimension: string, value: number): FieldError | undefined {
    if (!Number.isFinite(value)) return "invalid-observation";
    this.ensurePath(path, option, dimension);
    this.indexCoordinate(path, option, dimension);
    this.entry(path, option, dimension)!.cell.weight = value;
    return undefined;
  }

  /** @internal */ setGate(path: string[], option: string, dimension: string, viable: boolean): void {
    this.ensurePath(path, option, dimension);
    this.indexCoordinate(path, option, dimension);
    this.entry(path, option, dimension)!.cell.gate = viable;
  }

  /** @internal */ mergeEvidence(path: string[], option: string, dimension: string, evidence: Evidence): number | FieldError {
    if (
      !Number.isFinite(evidence.precision)
      || evidence.precision < 0
      || !Number.isFinite(evidence.information)
      || (evidence.precision === 0 && evidence.information !== 0)
      || !Number.isSafeInteger(evidence.observations)
      || evidence.observations < 0
    ) return "invalid-observation";
    const target = this.entry(path, option, dimension)?.cell.evidence ?? ZERO_EVIDENCE();
    if (!Number.isFinite(target.precision + evidence.precision)
      || !Number.isFinite(target.information + evidence.information)) return "invalid-observation";
    this.ensurePath(path, option, dimension);
    this.indexCoordinate(path, option, dimension);
    const stored = this.entry(path, option, dimension)!.cell.evidence;
    stored.precision += evidence.precision;
    stored.information += evidence.information;
    stored.observations = Math.min(Number.MAX_SAFE_INTEGER, stored.observations + evidence.observations);
    return this.recomputeUp(path, option, dimension);
  }

  /** @internal */ discountEvidence(
    path: string[],
    option: string,
    dimension: string,
    factor: number,
  ): DiscountTrace | FieldError {
    if (!Number.isFinite(factor) || factor < 0 || factor > 1) return "invalid-discount";
    const entry = this.entry(path, option, dimension);
    if (entry === undefined || entry.cell.evidence.observations <= 0) return "no-evidence";
    const precisionBefore = entry.cell.evidence.precision;
    entry.cell.evidence.precision *= factor;
    entry.cell.evidence.information *= factor;
    const updatedNodes = this.recomputeUp(path, option, dimension);
    return {
      address: this.addressForPath(path),
      option,
      dimension,
      factor,
      precision_before: precisionBefore,
      precision_after: entry.cell.evidence.precision,
      updated_nodes: updatedNodes,
    };
  }

  /** @internal */ compactSubtree(
    path: string[],
    option: string,
    dimension: string,
  ): CompactionTrace | FieldError {
    if (path.length === 0) return "root-compaction";
    const target = this.entry(path, option, dimension);
    if (target === undefined || target.cell.messageToParent.precision <= 0) return "no-evidence";
    const message = { ...target.cell.messageToParent };
    const compacted = [...this.cells.entries()].filter(([, entry]) =>
      entry.key.option === option
      && entry.key.dimension === dimension
      && path.every((value, index) => entry.key.path[index] === value)
    );
    const observations = compacted.reduce(
      (total, [, entry]) => Math.min(
        Number.MAX_SAFE_INTEGER,
        total + entry.cell.evidence.observations,
      ),
      0,
    );

    const parentPath = path.slice(0, -1);
    const parent = this.entry(parentPath, option, dimension)!;
    if (!Number.isFinite(parent.cell.evidence.precision + message.precision)
      || !Number.isFinite(parent.cell.evidence.information + message.information)) return "invalid-observation";
    parent.cell.childMessages = subNatural(parent.cell.childMessages, message);
    parent.cell.evidence.precision += message.precision;
    parent.cell.evidence.information += message.information;
    parent.cell.evidence.observations = Math.min(
      Number.MAX_SAFE_INTEGER,
      parent.cell.evidence.observations + observations,
    );

    for (const [id] of compacted) this.cells.delete(id);
    for (const [id, entries] of this.pathIndex) {
      const indexedPath = JSON.parse(id) as string[];
      if (path.every((value, index) => indexedPath[index] === value)) {
        entries.delete(JSON.stringify([option, dimension]));
        if (entries.size === 0) this.pathIndex.delete(id);
      }
    }
    for (const [id, dimensions] of this.stateDimensions) {
      const [indexedPath, indexedOption] = JSON.parse(id) as [string[], string];
      if (indexedOption === option && path.every((value, index) => indexedPath[index] === value)) {
        dimensions.delete(dimension);
        if (dimensions.size === 0) this.stateDimensions.delete(id);
      }
    }
    this.indexCoordinate(parentPath, option, dimension);
    this.processVarianceFrozen = true;
    const updatedNodes = this.recomputeUp(parentPath, option, dimension);
    return {
      address: this.addressForPath(path),
      compacted_into: this.addressForPath(parentPath),
      option,
      dimension,
      observations,
      removed_cells: compacted.length,
      updated_nodes: updatedNodes,
    };
  }

  private posteriorPath(
    path: string[],
    option: string,
    dimension: string,
    work: ResolveWork,
  ): ResidualTrace[] {
    work.parameter_lookups++;
    const root = this.entry([], option, dimension)?.cell;
    const rootPrior = root?.prior ?? 0;
    let external: Gaussian = { mean: rootPrior, variance: this.rootProcessVariance(dimension) };
    let posterior = combine(external, root === undefined ? ZERO_NATURAL() : this.subtree(root));
    const traces: ResidualTrace[] = [{
      address: {},
      posterior_mean: posterior.mean,
      posterior_variance: posterior.variance,
      residual_mean: posterior.mean,
      declared_prior: rootPrior,
      observations: root?.evidence.observations ?? 0,
    }];

    for (let depth = 0; depth < path.length; depth++) {
      work.parameter_lookups += 2;
      const parent = this.entry(path.slice(0, depth), option, dimension)?.cell;
      const child = this.entry(path.slice(0, depth + 1), option, dimension)?.cell;
      const selectedMessage = child?.messageToParent ?? ZERO_NATURAL();
      const outsideChild = parent === undefined
        ? ZERO_NATURAL()
        : subNatural(this.subtree(parent), selectedMessage);
      const parentCavity = combine(external, outsideChild);
      const childPrior = child?.prior ?? 0;
      external = {
        mean: parentCavity.mean + childPrior,
        variance: parentCavity.variance + this.processVariance(dimension, depth),
      };
      const previousMean = posterior.mean;
      posterior = combine(external, child === undefined ? ZERO_NATURAL() : this.subtree(child));
      traces.push({
        address: this.addressForPath(path.slice(0, depth + 1)),
        posterior_mean: posterior.mean,
        posterior_variance: posterior.variance,
        residual_mean: posterior.mean - previousMean,
        declared_prior: childPrior,
        observations: child?.evidence.observations ?? 0,
      });
    }
    return traces;
  }

  private candidatesOnPath(path: string[], work: ResolveWork): Map<string, Set<string>> {
    const candidates = new Map<string, Set<string>>();
    for (let depth = 0; depth <= path.length; depth++) {
      work.address_nodes++;
      const bucket = this.pathIndex.get(this.pathId(path.slice(0, depth)));
      if (bucket === undefined) continue;
      work.stored_cells_visited += bucket.size;
      for (const [option, dimension] of bucket.values()) {
        if (option === FIELD_SHARED) continue;
        const dimensions = candidates.get(option) ?? new Set<string>();
        dimensions.add(dimension);
        candidates.set(option, dimensions);
      }
    }
    const options = [...candidates.keys()];
    for (let depth = 0; depth <= path.length; depth++) {
      const prefix = path.slice(0, depth);
      for (const option of options) {
        work.parameter_lookups++;
        const dimensions = this.stateDimensions.get(this.stateDimensionId(prefix, option));
        if (dimensions === undefined) continue;
        work.stored_cells_visited += dimensions.size;
        const target = candidates.get(option)!;
        for (const dimension of dimensions) target.add(dimension);
      }
    }
    return candidates;
  }

  /**
   * Discover only explicitly allowed candidates without scanning unrelated
   * options stored at shared ancestors. An option still has to be declared on
   * the addressed path; the allow-list restricts a decision, it does not
   * manufacture an otherwise absent candidate.
   */
  private candidatesAmong(
    path: string[],
    options: Iterable<string>,
    work: ResolveWork,
  ): Map<string, Set<string>> {
    const allowed = [...new Set(options)].sort(compareStrings);
    const candidates = new Map<string, Set<string>>();
    for (let depth = 0; depth <= path.length; depth++) {
      work.address_nodes++;
      const prefix = path.slice(0, depth);
      const bucket = this.pathIndex.get(this.pathId(prefix));
      if (bucket === undefined) continue;
      for (const option of allowed) {
        work.parameter_lookups++;
        const dimensions = this.stateDimensions.get(this.stateDimensionId(prefix, option));
        if (dimensions === undefined) continue;
        for (const dimension of dimensions) {
          work.parameter_lookups++;
          if (!bucket.has(JSON.stringify([option, dimension]))) continue;
          work.stored_cells_visited++;
          const target = candidates.get(option) ?? new Set<string>();
          target.add(dimension);
          candidates.set(option, target);
        }
      }
    }
    for (let depth = 0; depth <= path.length; depth++) {
      const prefix = path.slice(0, depth);
      for (const [option, target] of candidates) {
        work.parameter_lookups++;
        const dimensions = this.stateDimensions.get(this.stateDimensionId(prefix, option));
        if (dimensions === undefined) continue;
        work.stored_cells_visited += dimensions.size;
        for (const dimension of dimensions) target.add(dimension);
      }
    }
    return candidates;
  }

  private weightOnPath(path: string[], option: string, dimension: string, work: ResolveWork): number {
    let sum = 0;
    let found = false;
    for (let depth = 0; depth <= path.length; depth++) {
      for (const candidate of [FIELD_SHARED, option]) {
        work.parameter_lookups++;
        const weight = this.entry(path.slice(0, depth), candidate, dimension)?.cell.weight;
        if (weight !== undefined) {
          sum += weight;
          found = true;
        }
      }
    }
    return found ? sum : 1;
  }

  private gateOnPath(path: string[], option: string, dimension: string, work: ResolveWork): boolean {
    for (let depth = 0; depth <= path.length; depth++) {
      work.parameter_lookups++;
      if (this.entry(path.slice(0, depth), option, dimension)?.cell.gate === false) return false;
    }
    return true;
  }

  private numericOnPath(path: string[], option: string, dimension: string, work: ResolveWork): boolean {
    for (let depth = 0; depth <= path.length; depth++) {
      work.parameter_lookups++;
      const cell = this.entry(path.slice(0, depth), option, dimension)?.cell;
      if (cell !== undefined
        && (cell.prior !== undefined
          || cell.evidence.precision > 0
          || cell.evidence.observations > 0
          || cell.childMessages.precision > 0)) return true;
    }
    return false;
  }

  private decideCandidates(
    path: string[],
    requestedPolicy: DecisionPolicy,
    candidates: Map<string, Set<string>>,
    work: ResolveWork,
  ): FieldDecision {
    const temperature = Number.isFinite(requestedPolicy.temperature) ? Math.max(0, requestedPolicy.temperature) : 0;
    const policy = { temperature };
    const alternatives: FieldResolvedOption[] = [];
    for (const [option, dimensions] of candidates) {
      let viable = true;
      let expectedScore = 0;
      let scoreVariance = 0;
      const traces: FieldDimensionTrace[] = [];
      for (const dimension of [...dimensions].sort(compareStrings)) {
        if (!this.gateOnPath(path, option, dimension, work)) viable = false;
        const numeric = this.numericOnPath(path, option, dimension, work);
        const contributions = numeric ? this.posteriorPath(path, option, dimension, work) : [];
        const last = contributions.at(-1);
        const mean = last?.posterior_mean ?? 0;
        const variance = last?.posterior_variance ?? 0;
        const weight = numeric ? this.weightOnPath(path, option, dimension, work) : 0;
        const contribution = mean * weight;
        expectedScore += contribution;
        scoreVariance += variance * weight * weight;
        traces.push({ dimension, mean, variance, weight, contribution, contributions });
      }
      alternatives.push({
        option,
        viable,
        expected_score: expectedScore,
        score_variance: scoreVariance,
        decision_score: expectedScore - temperature * Math.sqrt(scoreVariance),
        dimensions: traces,
      });
    }
    alternatives.sort((left, right) =>
      Number(right.viable) - Number(left.viable)
      || left.decision_score - right.decision_score
      || compareStrings(left.option, right.option)
    );
    work.ranked_options = alternatives.length;
    return { address: this.addressForPath(path), policy, alternatives, work };
  }

  decide(input: Address, requestedPolicy: DecisionPolicy = { temperature: 0 }): FieldDecision | FieldError {
    const path = this.path(input);
    if (isFieldError(path)) return path;
    const work: ResolveWork = { address_nodes: 0, stored_cells_visited: 0, parameter_lookups: 0, ranked_options: 0 };
    return this.decideCandidates(path, requestedPolicy, this.candidatesOnPath(path, work), work);
  }

  /** Resolve only options allowed by the current host/product configuration. */
  decideAmong(
    input: Address,
    options: Iterable<string>,
    requestedPolicy: DecisionPolicy = { temperature: 0 },
  ): FieldDecision | FieldError {
    const path = this.path(input);
    if (isFieldError(path)) return path;
    const work: ResolveWork = { address_nodes: 0, stored_cells_visited: 0, parameter_lookups: 0, ranked_options: 0 };
    return this.decideCandidates(path, requestedPolicy, this.candidatesAmong(path, options, work), work);
  }

  resolve(input: Address, policy: DecisionPolicy = { temperature: 0 }): string[] | FieldError {
    const decision = this.decide(input, policy);
    return isFieldError(decision)
      ? decision
      : decision.alternatives.filter(({ viable }) => viable).map(({ option }) => option);
  }

  static receipt(decision: FieldDecision, tensorRevision: string, selected: string[]): RoutingReceipt {
    return { tensor_revision: tensorRevision, decision, selected: [...selected] };
  }

  cells_(): JsonFieldCell[] {
    const cells = [...this.cells.values()]
      .filter(({ cell }) => cell.prior !== undefined
        || cell.weight !== undefined
        || cell.gate !== undefined
        || cell.evidence.observations !== 0
        || cell.evidence.precision !== 0
        || cell.evidence.information !== 0)
      .map(({ key, cell }) => ({
        address: this.addressForPath(key.path),
        option: key.option,
        dimension: key.dimension,
        prior: cell.prior,
        weight: cell.weight,
        gate: cell.gate,
        evidence: { ...cell.evidence },
      }));
    cells.sort((left, right) => compareStrings(
      JSON.stringify([Object.values(left.address), left.option, left.dimension]),
      JSON.stringify([Object.values(right.address), right.option, right.dimension]),
    ));
    return cells;
  }

  /** Number of materialized parameter nodes, including cached path prefixes. */
  storedCellCount(): number {
    return this.cells.size;
  }

  toJson(): string {
    return JSON.stringify({
      layers: this.layers,
      default_root_process_variance: this.defaultRootProcessVariance,
      default_process_variance: this.defaultProcessVariance,
      dimension_root_process_variance: Object.fromEntries([...this.dimensionRootProcessVariance].sort(([a], [b]) => compareStrings(a, b))),
      dimension_process_variance: Object.fromEntries([...this.dimensionProcessVariance].sort(([a], [b]) => compareStrings(a, b))),
      process_variance_frozen: this.processVarianceFrozen,
      cells: this.cells_().map((cell) => {
        const encoded: Record<string, unknown> = {
          address: cell.address,
          option: cell.option,
          dimension: cell.dimension,
        };
        if (cell.prior !== undefined) encoded.prior = cell.prior;
        if (cell.weight !== undefined) encoded.weight = cell.weight;
        if (cell.gate !== undefined) encoded.gate = cell.gate;
        if (cell.evidence.observations !== 0 || cell.evidence.precision !== 0 || cell.evidence.information !== 0) {
          encoded.evidence = cell.evidence;
        }
        return encoded;
      }),
    });
  }

  static fromJson(snapshot: string): AddressedField | undefined {
    let raw: unknown;
    try { raw = JSON.parse(snapshot); } catch { return undefined; }
    if (raw === null || typeof raw !== "object") return undefined;
    const object = raw as Record<string, unknown>;
    if (!Array.isArray(object.layers)
      || !object.layers.every((value) => typeof value === "string" && value.length > 0)
      || new Set(object.layers).size !== object.layers.length) return undefined;
    const defaults = object.default_process_variance;
    if (!Array.isArray(defaults)
      || defaults.length !== object.layers.length
      || !defaults.every((value) => typeof value === "number" && Number.isFinite(value) && value > 0)) return undefined;
    const field = new AddressedField(object.layers as string[]);
    if (object.default_root_process_variance !== undefined
      && (typeof object.default_root_process_variance !== "number"
        || !Number.isFinite(object.default_root_process_variance)
        || object.default_root_process_variance <= 0)) return undefined;
    field.defaultRootProcessVariance = (object.default_root_process_variance as number | undefined) ?? 1;
    field.defaultProcessVariance = [...defaults] as number[];
    if (object.dimension_root_process_variance !== undefined) {
      if (object.dimension_root_process_variance === null || typeof object.dimension_root_process_variance !== "object") return undefined;
      for (const [dimension, value] of Object.entries(object.dimension_root_process_variance)) {
        if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return undefined;
        field.dimensionRootProcessVariance.set(dimension, value);
      }
    }
    if (object.dimension_process_variance !== undefined) {
      if (object.dimension_process_variance === null || typeof object.dimension_process_variance !== "object") return undefined;
      for (const [dimension, value] of Object.entries(object.dimension_process_variance)) {
        if (!Array.isArray(value)
          || value.length !== field.layers.length
          || !value.every((item) => typeof item === "number" && Number.isFinite(item) && item > 0)) return undefined;
        field.dimensionProcessVariance.set(dimension, [...value]);
      }
    }
    if (!Array.isArray(object.cells)) return undefined;
    if (object.process_variance_frozen !== undefined
      && typeof object.process_variance_frozen !== "boolean") return undefined;
    field.processVarianceFrozen = object.process_variance_frozen === true;
    const seen = new Set<string>();
    for (const rawCell of object.cells) {
      if (rawCell === null || typeof rawCell !== "object") return undefined;
      const cell = rawCell as Record<string, unknown>;
      if (cell.address === null
        || typeof cell.address !== "object"
        || Array.isArray(cell.address)
        || !Object.values(cell.address).every((value) => typeof value === "string")
        || typeof cell.option !== "string"
        || typeof cell.dimension !== "string") return undefined;
      const path = field.path(cell.address as Address);
      if (isFieldError(path)) return undefined;
      const evidenceRaw = cell.evidence ?? { precision: 0, information: 0, observations: 0 };
      if (evidenceRaw === null || typeof evidenceRaw !== "object") return undefined;
      const evidence = evidenceRaw as Record<string, unknown>;
      if (typeof evidence.precision !== "number"
        || typeof evidence.information !== "number"
        || typeof evidence.observations !== "number"
        || !Number.isFinite(evidence.precision)
        || evidence.precision < 0
        || !Number.isFinite(evidence.information)
        || (evidence.precision === 0 && evidence.information !== 0)
        || !Number.isSafeInteger(evidence.observations)
        || evidence.observations < 0
        || (cell.prior !== undefined && (typeof cell.prior !== "number" || !Number.isFinite(cell.prior)))
        || (cell.weight !== undefined && (typeof cell.weight !== "number" || !Number.isFinite(cell.weight)))) return undefined;
      const id = field.keyId(path, cell.option, cell.dimension);
      if (seen.has(id)) return undefined;
      seen.add(id);
      field.ensurePath(path, cell.option, cell.dimension);
      field.indexCoordinate(path, cell.option, cell.dimension);
      const target = field.entry(path, cell.option, cell.dimension)!.cell;
      if (cell.gate !== undefined && typeof cell.gate !== "boolean") return undefined;
      target.prior = cell.prior as number | undefined;
      target.weight = cell.weight as number | undefined;
      target.gate = cell.gate as boolean | undefined;
      target.evidence = evidence as unknown as Evidence;
    }
    field.rebuildMessages();
    return field;
  }

  /** @internal */ posteriorFor(path: string[], option: string, dimension: string): [number, number] | undefined {
    const last = this.posteriorPath(path, option, dimension, {
      address_nodes: 0,
      stored_cells_visited: 0,
      parameter_lookups: 0,
      ranked_options: 0,
    }).at(-1);
    return last === undefined ? undefined : [last.posterior_mean, last.posterior_variance];
  }

  /** Exact compressed Gaussian message this addressed subtree sends upward. */
  messageToParent(input: Address, option: string, dimension: string): Natural | FieldError {
    const path = this.path(input);
    if (isFieldError(path)) return path;
    if (path.length === 0) return "address-too-shallow";
    return { ...(this.entry(path, option, dimension)?.cell.messageToParent ?? ZERO_NATURAL()) };
  }
}

export class FieldWriter {
  constructor(private readonly field: AddressedField, private readonly floor: number | undefined) {}

  private checkedPath(input: Address): string[] | FieldError {
    const path = this.field.path(input);
    if (isFieldError(path)) return path;
    if (this.floor === undefined) return path;
    return path.length === 0 || path.length - 1 < this.floor ? "write-up" : path;
  }

  setPrior(input: Address, option: string, dimension: string, value: number): FieldError | undefined {
    const path = this.checkedPath(input);
    return isFieldError(path) ? path : this.field.setPrior(path, option, dimension, value);
  }

  setWeight(input: Address, dimension: string, value: number): FieldError | undefined {
    const path = this.checkedPath(input);
    return isFieldError(path) ? path : this.field.setWeight(path, FIELD_SHARED, dimension, value);
  }

  setOptionWeight(input: Address, option: string, dimension: string, value: number): FieldError | undefined {
    const path = this.checkedPath(input);
    return isFieldError(path) ? path : this.field.setWeight(path, option, dimension, value);
  }

  setGate(input: Address, option: string, dimension: string, viable: boolean): FieldError | undefined {
    const path = this.checkedPath(input);
    if (isFieldError(path)) return path;
    this.field.setGate(path, option, dimension, viable);
    return undefined;
  }

  observe(
    input: Address,
    option: string,
    dimension: string,
    value: number,
    variance: number,
  ): FeedbackTrace | FieldError {
    const path = this.checkedPath(input);
    if (isFieldError(path)) return path;
    if (!Number.isFinite(value)) return "invalid-observation";
    if (!Number.isFinite(variance) || variance <= 0) return "invalid-variance";
    const before = this.field.posteriorFor(path, option, dimension);
    const precision = 1 / variance;
    const updated = this.field.mergeEvidence(path, option, dimension, {
      precision,
      information: value * precision,
      observations: 1,
    });
    if (isFieldError(updated)) return updated;
    const after = this.field.posteriorFor(path, option, dimension)!;
    return {
      address: address(input),
      option,
      dimension,
      observed: value,
      observation_variance: variance,
      updated_nodes: updated,
      posterior_before: before,
      posterior_after: after,
    };
  }

  mergeEvidence(input: Address, option: string, dimension: string, evidence: Evidence): number | FieldError {
    const path = this.checkedPath(input);
    return isFieldError(path) ? path : this.field.mergeEvidence(path, option, dimension, evidence);
  }

  /** Reduce evidence precision by a product-supplied clock-derived factor. */
  discountEvidence(
    input: Address,
    option: string,
    dimension: string,
    factor: number,
  ): DiscountTrace | FieldError {
    const path = this.checkedPath(input);
    return isFieldError(path)
      ? path
      : this.field.discountEvidence(path, option, dimension, factor);
  }

  /**
   * Collapse a retired subtree into its parent's sufficient statistics. This is
   * an explicit integrity upgrade, so the writer must control the parent.
   */
  compactSubtree(
    input: Address,
    option: string,
    dimension: string,
  ): CompactionTrace | FieldError {
    const path = this.field.path(input);
    if (isFieldError(path)) return path;
    if (path.length === 0) return "root-compaction";
    if (this.floor !== undefined && path.length - 2 < this.floor) return "write-up";
    return this.field.compactSubtree(path, option, dimension);
  }
}
