// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * sporewright — one object, a sparse tensor `T[level][instance][option][dimension]`,
 * and the operations on it. Byte-equivalent with the Rust crate. See docs/MODEL.md.
 */
export * from "./tensor";

export * from "./sync";

export * from "./normalize";

export * from "./budget";

export * from "./config";

export * from "./persist";

export * from "./trust";
