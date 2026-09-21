// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Persistence — the durability/versioning port. Mirrors the Rust `persist.rs`.
 *
 * The tensor core is clock-free; the database is the clock. This port is where a
 * product's facts live and where versioning lives — an edge KV or a relational DB,
 * SQLite/a file, IndexedDB, or memory. A real impl stores {@link Tensor.cells_} as
 * versioned rows and answers "everything since version N"; the client overlays the
 * returned cells with {@link Tensor.applyJson}. {@link MemStore} is the trivial
 * reference: a whole-snapshot string, no versioning.
 */

/** The persistence port: load and save a serialized tensor (`Tensor.toJson`). */
export interface Store {
  load(): string | undefined;
  save(snapshot: string): void;
}

/** An in-memory {@link Store} — the reference impl. Real durability + versioning is
 *  the product's database. */
export class MemStore implements Store {
  private snap: string | undefined;
  load(): string | undefined {
    return this.snap;
  }
  save(snapshot: string): void {
    this.snap = snapshot;
  }
}
