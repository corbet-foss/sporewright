// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden normalization vectors, shared with the TS core.
//!
//! Each vector in `../../tests/norm-vectors/*.json` pins one normalized routing
//! decision: a `{levels, cells}` base tensor of RAW per-option value cells, a
//! `norm` spec (the write level, a per-dim `scales` map, and the per-dim raw MRS
//! `weights` to derive), a read `cursor`, and the `expected` resolved order AFTER
//! the normalizer's effective weights are written.
//!
//! This test rebuilds the tensor via the persistence-port wire
//! (`Tensor::from_json`), then for each dim writes `set_weight(dim,
//! Norm::effective_weight(dim, raw_weight))`, resolves at the cursor, and asserts
//! the order.
//!
//! A vector may instead carry a `valueNorm` block (the VALUE-side route): then the
//! runner rewrites each raw `(option, dim)` value as `Norm::normalize_value(raw, lo,
//! hi)` (per-dim `ranges`), writes the shared `weights`, resolves, and asserts BOTH
//! the per-cell `expectedValues` (bit-exact), the `expectedGated` options (dropped by
//! resolve), AND the `expected` order. This is the only vector that drives the
//! value-side min-max route the docs call "decision-equivalent across cores"; the
//! `norm` (weight-side) vectors drive only `effective_weight`.
//!
//! The SAME files are read by
//! `packages/sporewright/src/norm-vectors.test.ts`, making them a cross-core
//! decision-equivalence guard for the normalizer: both cores derive the identical
//! effective weights / normalized values, write them through the unchanged
//! `resolve`, and MUST return the identical order. The goldens were captured from
//! the live normalized `resolve` — never hand-fabricated.

use serde_json::Value;
use sporewright::{Cursor, Norm, Tensor};
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/norm-vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("norm-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/norm-vectors dir must exist") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("vector {path:?} is not valid JSON: {e}"));
        out.push((path.file_name().unwrap().to_string_lossy().into_owned(), v));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!out.is_empty(), "no norm vectors found in {dir:?}");
    out
}

fn cursor_from(v: &Value) -> Cursor {
    let mut c = Cursor::new();
    if let Some(obj) = v.get("cursor").and_then(Value::as_object) {
        for (k, val) in obj {
            c.insert(k.clone(), val.as_str().unwrap().to_string());
        }
    }
    c
}

fn expected_from(v: &Value) -> Vec<String> {
    v.get("expected")
        .and_then(Value::as_array)
        .expect("vector must have an `expected` array")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

fn run_weight_vector(file: &str, vector: &Value) {
    let tensor_json = vector.get("tensor").unwrap().to_string();
    let mut t = Tensor::from_json(&tensor_json)
        .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));

    let norm = vector.get("norm").unwrap();
    let write_level = norm.get("writeLevel").and_then(Value::as_str).unwrap();
    let scales = norm
        .get("scales")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("{file}: missing `norm.scales`"));
    let weights = norm
        .get("weights")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("{file}: missing `norm.weights`"));

    // Build the normalizer from the per-dim scales.
    let mut n = Norm::new();
    for (dim, s) in scales {
        n = n.with_scale(dim, s.as_f64().unwrap());
    }

    // Write each dim's EFFECTIVE weight (raw MRS / scale) — the only difference
    // from a raw consumer. `resolve` itself is unchanged.
    {
        let mut w = t.writer(write_level).unwrap();
        for (dim, raw) in weights {
            let eff = n.effective_weight(dim, raw.as_f64().unwrap());
            w.set_weight(write_level, "", dim, eff)
                .unwrap_or_else(|e| panic!("{file}: set_weight failed: {e:?}"));
        }
    }

    let cursor = cursor_from(vector);
    let expected = expected_from(vector);
    let resolved = t.resolve(&cursor);
    assert_eq!(
        resolved, expected,
        "{file}: normalized resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
        vector.get("description").and_then(Value::as_str).unwrap_or("")
    );
}

/// VALUE-side route: rewrite each raw `(option, dim)` value as
/// `Norm::normalize_value(raw, lo, hi)`, write the shared weights, resolve, and assert
/// the per-cell normalized values, the gated (dropped) options, and the order.
fn run_value_vector(file: &str, vector: &Value) {
    let tensor_json = vector.get("tensor").unwrap().to_string();
    let mut t = Tensor::from_json(&tensor_json)
        .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));

    let spec = vector.get("valueNorm").unwrap();
    let write_level = spec.get("writeLevel").and_then(Value::as_str).unwrap();
    let ranges = spec.get("ranges").and_then(Value::as_object).unwrap();
    let weights = spec.get("weights").and_then(Value::as_object).unwrap();

    let cursor = cursor_from(vector);
    let options = t.options(&cursor);

    // First collect the normalized writes immutably (a writer borrows the tensor
    // mutably, so we cannot read raw values while one is open). `normalize_value`
    // passes a non-finite gate through untouched, so a gated value stays a gate.
    let mut norm_writes: Vec<(String, String, f64)> = Vec::new();
    for (dim, range) in ranges {
        let lo = range.get("lo").and_then(Value::as_f64).unwrap();
        let hi = range.get("hi").and_then(Value::as_f64).unwrap();
        for opt in &options {
            if let Some(raw) = t.value(&cursor, opt, dim) {
                let norm = Norm::normalize_value(raw, lo, hi);
                norm_writes.push((opt.clone(), dim.clone(), norm));
            }
        }
    }

    {
        let mut w = t.writer(write_level).unwrap();
        for (opt, dim, norm) in &norm_writes {
            w.set_value(write_level, "", opt, dim, *norm)
                .unwrap_or_else(|e| panic!("{file}: set_value failed: {e:?}"));
        }
        for (dim, raw) in weights {
            w.set_weight(write_level, "", dim, raw.as_f64().unwrap())
                .unwrap_or_else(|e| panic!("{file}: set_weight failed: {e:?}"));
        }
    }

    for want in spec_array(vector, "expectedValues") {
        let opt = want.get("option").and_then(Value::as_str).unwrap();
        let dim = want.get("dim").and_then(Value::as_str).unwrap();
        let value = want.get("value").and_then(Value::as_f64).unwrap();
        let got = t.value(&cursor, opt, dim);
        assert_eq!(
            got,
            Some(value),
            "{file}: normalized value of ({opt}, {dim}) changed. expected {value}, got {got:?}"
        );
    }
    for gated in spec_array(vector, "expectedGated") {
        let opt = gated.get("option").and_then(Value::as_str).unwrap();
        let resolved = t.resolve(&cursor);
        assert!(
            !resolved.contains(&opt.to_string()),
            "{file}: option `{opt}` carries a gate (non-finite value) and MUST be dropped, but appears in {resolved:?}"
        );
    }

    let expected = expected_from(vector);
    let resolved = t.resolve(&cursor);
    assert_eq!(
        resolved, expected,
        "{file}: value-normalized resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
        vector.get("description").and_then(Value::as_str).unwrap_or("")
    );
}

fn spec_array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

#[test]
fn golden_norm_vectors_pin_the_normalized_decision() {
    for (file, vector) in load_vectors() {
        if vector.get("valueNorm").is_some() {
            run_value_vector(&file, &vector);
        } else {
            run_weight_vector(&file, &vector);
        }
    }
}
