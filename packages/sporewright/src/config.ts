// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/** Declarative construction of a stacked tensor. */

import { SHARED, Tensor, type Scope, type Value, type WriteError } from "./tensor";

export type SeedKind = "value" | "bytes" | "weight" | "option-weight";

export interface SeedCell {
  floor: string;
  layer: string;
  scope: Scope;
  option: string;
  dim: string;
  kind: SeedKind;
  value: Value;
}

export interface TensorConfig {
  layers: string[];
  seeds: SeedCell[];
}

export function valueCell(
  floor: string,
  layer: string,
  cellScope: Scope,
  option: string,
  dim: string,
  value: number,
): SeedCell {
  return { floor, layer, scope: { ...cellScope }, option, dim, kind: "value", value: { f64: value } };
}

export function bytesCell(
  floor: string,
  layer: string,
  cellScope: Scope,
  option: string,
  dim: string,
  value: Uint8Array,
): SeedCell {
  return { floor, layer, scope: { ...cellScope }, option, dim, kind: "bytes", value: { bytes: value } };
}

export function weightCell(
  floor: string,
  layer: string,
  cellScope: Scope,
  dim: string,
  value: number,
): SeedCell {
  return { floor, layer, scope: { ...cellScope }, option: SHARED, dim, kind: "weight", value: { f64: value } };
}

export function optionWeightCell(
  floor: string,
  layer: string,
  cellScope: Scope,
  option: string,
  dim: string,
  value: number,
): SeedCell {
  return { floor, layer, scope: { ...cellScope }, option, dim, kind: "option-weight", value: { f64: value } };
}

export function tensorConfig(layers: readonly string[], seeds: readonly SeedCell[] = []): TensorConfig {
  return { layers: [...layers], seeds: [...seeds] };
}

export function withCell(config: TensorConfig, cell: SeedCell): TensorConfig {
  return { layers: [...config.layers], seeds: [...config.seeds, cell] };
}

export function instantiate(config: TensorConfig): Tensor | WriteError {
  const tensor = new Tensor(config.layers);
  for (const cell of config.seeds) {
    const writer = tensor.writer(cell.floor);
    if (writer === undefined) return "unknown-layer";
    let error: WriteError | undefined;
    if (cell.kind === "value") {
      if (!("f64" in cell.value)) return "bad-value-kind";
      error = writer.setValue(cell.layer, cell.scope, cell.option, cell.dim, cell.value.f64);
    } else if (cell.kind === "bytes") {
      if (!("bytes" in cell.value)) return "bad-value-kind";
      error = writer.setBytes(cell.layer, cell.scope, cell.option, cell.dim, cell.value.bytes);
    } else if (cell.kind === "weight") {
      if (!("f64" in cell.value)) return "bad-value-kind";
      error = writer.setWeight(cell.layer, cell.scope, cell.dim, cell.value.f64);
    } else {
      if (!("f64" in cell.value)) return "bad-value-kind";
      error = writer.setOptionWeight(cell.layer, cell.scope, cell.option, cell.dim, cell.value.f64);
    }
    if (error !== undefined) return error;
  }
  return tensor;
}
