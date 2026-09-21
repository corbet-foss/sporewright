// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Mesh roles — the thin device side of the tensor. Mirrors the Rust `sync.rs`.
 *
 * Sync itself is not here: state moves `device → API → DB` and back as cells
 * (`Tensor.cells_`/`applyJson`), versioned by the DB, carried by a lease/poll
 * transport port. This module is the device operations on top of a {@link Writer}:
 * **observe** a measured value, **gate** an option's reachability, and **attest** a
 * subject's capability (the input `corroborate` rolls up).
 *
 * Peer-as-option: a reachable peer is an option (`"peer:A"`); its reachability is
 * the {@link REACH} gate — `Infinity` when unreachable (dropped by `resolve`),
 * `0`/absent when reachable. Work routes P2P; state syncs via the orchestrator star.
 */
import type { Tensor, WriteError, Writer } from "./tensor";

/** The reachability gate dimension for a peer option. */
export const REACH = "reach";

/** The capability-gate dimension prefix: a gate on `priv:<cap>` carries `Infinity`
 *  when an option **lacks** a required capability `<cap>` (dropped by `resolve`);
 *  absent or `0` ⇒ the option satisfies it. The `<cap>` vocabulary
 *  (`ip:residential`, `gpu:burst`, `browser:full`, `geo:CH`, …) is the product's —
 *  the engine never learns a single capability name. Mirrors Rust `GATE_PREFIX`. */
export const GATE_PREFIX = "priv:";

/** Compose the gate dimension `priv:<cap>` for a capability `cap`. */
export function gateDim(cap: string): string {
  return `${GATE_PREFIX}${cap}`;
}

/** **gateCapabilities** — the vocabulary-agnostic capability gate. For each `option`
 *  that the `offers` predicate says does **not** provide a `required` capability,
 *  write `Infinity` on `priv:<cap>` at `(level, inst)`, so `resolve` drops that option
 *  for this slice. Options offering every required capability are left untouched (no
 *  gate ⇒ still candidates, ordered by their judgement dims).
 *
 *  This is the option-axis feasibility primitive of MODEL.md §3(C): the indicator of
 *  the feasible set, `0`/absent when allowed, `Infinity` when not. It is **purely
 *  additive** (only ever adds gates, never clears them) and knows nothing about any
 *  specific capability — the `<cap>` strings are the product's vocabulary. Decision-
 *  equivalent with Rust `gate_capabilities`.
 *
 *  Returns the first {@link WriteError} (write-up or unknown level) hit, or `undefined`
 *  when every gate landed. */
export function gateCapabilities(
  writer: Writer,
  level: string,
  inst: string,
  options: readonly string[],
  required: readonly string[],
  offers: (option: string, cap: string) => boolean,
): WriteError | undefined {
  for (const cap of required) {
    const dim = gateDim(cap);
    for (const option of options) {
      if (!offers(option, cap)) {
        const e = writer.setValue(level, inst, option, dim, Infinity);
        if (e) return e;
      }
    }
  }
  return undefined;
}

/** A device in the mesh: a {@link Writer} pinned to its own slice, with the device
 *  operations. It owns exactly its `(level, inst)` slice, so devices never conflict;
 *  the orchestrator then `reduce`s observations and `corroborate`s attestations. */
export class Device {
  private constructor(
    private w: Writer,
    private level: string,
    private inst: string,
  ) {}

  /** A device handle at `level` for instance `inst`; undefined if level unknown. */
  static at(t: Tensor, level: string, inst: string): Device | undefined {
    const w = t.writer(level);
    return w ? new Device(w, level, inst) : undefined;
  }

  /** Record a measured value for an `option` on this device's slice. */
  observe(option: string, dim: string, value: number): WriteError | undefined {
    return this.w.setValue(this.level, this.inst, option, dim, value);
  }

  /** Set whether an `option` is reachable now: `false` gates it (`REACH = Infinity`),
   *  `true` opens it (`0`). */
  gate(option: string, reachable: boolean): WriteError | undefined {
    return this.w.setValue(this.level, this.inst, option, REACH, reachable ? 0 : Infinity);
  }

  /** Attest whether a subject `option` has capability `dim` — the boolean input
   *  `corroborate` counts (independent reporters, self-excluded). */
  attest(option: string, dim: string, can: boolean): WriteError | undefined {
    return this.w.setValue(this.level, this.inst, option, dim, can ? 1.0 : 0.0);
  }
}
