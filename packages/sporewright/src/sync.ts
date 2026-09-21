// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/** Participant-side operations over a scoped preference layer. */

import type { PreferenceChange, Scope, Tensor, WriteError, Writer } from "./tensor";

export const REACH = "reach";
export const GATE_PREFIX = "priv:";

export function gateDim(capability: string): string {
  return `${GATE_PREFIX}${capability}`;
}

export function gateCapabilities(
  writer: Writer,
  layer: string,
  cellScope: Scope,
  options: readonly string[],
  required: readonly string[],
  offers: (option: string, capability: string) => boolean,
): WriteError | undefined {
  for (const capability of required) {
    const dimension = gateDim(capability);
    for (const option of options) {
      if (!offers(option, capability)) {
        const error = writer.setValue(layer, cellScope, option, dimension, Infinity);
        if (error !== undefined) return error;
      }
    }
  }
  return undefined;
}

/** A participant pinned to one exact layer-owned preference scope. */
export class Device {
  private constructor(
    private readonly writer: Writer,
    private readonly layer: string,
    private readonly cellScope: Scope,
  ) {}

  static at(tensor: Tensor, layer: string, cellScope: Scope): Device | undefined {
    const writer = tensor.writer(layer);
    return writer === undefined ? undefined : new Device(writer, layer, { ...cellScope });
  }

  observe(option: string, dimension: string, value: number): WriteError | undefined {
    return this.writer.setValue(this.layer, this.cellScope, option, dimension, value);
  }

  revise(
    option: string,
    dimension: string,
    observation: number,
    rate: number,
  ): PreferenceChange | WriteError {
    return this.writer.nudgeValue(this.layer, this.cellScope, option, dimension, observation, rate);
  }

  gate(option: string, reachable: boolean): WriteError | undefined {
    return this.writer.setValue(this.layer, this.cellScope, option, REACH, reachable ? 0 : Infinity);
  }

  attest(option: string, dimension: string, capable: boolean): WriteError | undefined {
    return this.writer.setValue(this.layer, this.cellScope, option, dimension, capable ? 1 : 0);
  }
}
