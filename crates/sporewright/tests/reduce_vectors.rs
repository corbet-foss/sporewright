// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden `reduce_median` / `roll_up` vectors, shared with the
//! TS core.
//!
//! Each vector in `../../tests/reduce-vectors/*.json` pins one decision of the
//! certified integrity up-path (MODEL.md §3): the median value-fold. Before this
//! corpus, `reduce_median` was pinned only by independent per-core unit literals
//! (both hardcode `0.30000000000000004`) that could drift apart silently — exactly
//! the even-median index arithmetic and finite filter a refactor touches. These are
//! the first SHARED vectors both cores read.
//!
//! Each vector has a `feeder` spec: the tensor `levels`, a `fromLevel`/`toLevel`/
//! `toInst`, and a list of raw per-instance value `writes` `{inst, option, dim,
//! value}` in a SCRAMBLED instance order (load-bearing: it must NOT match a sorted
//! scan, so the value-sort inside `reduce_median` is actually under test). The
//! runner builds a fresh feeder tensor, writes the values, runs `roll_up`
//! (`reduce_median` for every `(option, dim)` present), then asserts each
//! `expected` `{option, dim, value}` reads back BIT-EXACTLY at the cursor.
//!
//! The SAME files are read by `packages/sporewright/src/reduce-vectors.test.ts`,
//! making them a cross-core decision-equivalence guard: both cores roll the
//! identical raw cells up and MUST materialize the identical median. A `value` of
//! the string `"inf"`/`"-inf"`/`"nan"` writes the matching non-finite sentinel (so
//! the finite filter is exercised). The goldens were captured from the live
//! `reduce_median` — never hand-fabricated.

use serde_json::Value;
use sporewright::{scope, Cursor, Scope, Tensor};
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/reduce-vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("reduce-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/reduce-vectors dir must exist") {
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
    assert!(!out.is_empty(), "no reduce vectors found in {dir:?}");
    out
}

/// Parse a JSON value-field into an f64, honoring the wire sentinels for non-finite
/// values (so a `"value": "inf"` write exercises the finite filter).
fn parse_value(v: &Value) -> f64 {
    if let Some(f) = v.as_f64() {
        return f;
    }
    match v.as_str() {
        Some("inf") => f64::INFINITY,
        Some("-inf") => f64::NEG_INFINITY,
        Some("nan") => f64::NAN,
        other => panic!("unrecognized value sentinel {other:?}"),
    }
}

#[test]
fn golden_reduce_vectors_pin_the_median_rollup() {
    for (file, vector) in load_vectors() {
        assert_eq!(
            vector.get("kind").and_then(Value::as_str),
            Some("median"),
            "{file}: only the `median` reduce-vector kind is implemented"
        );

        let feeder_spec = vector
            .get("feeder")
            .unwrap_or_else(|| panic!("{file}: missing `feeder`"));
        let levels: Vec<String> = feeder_spec
            .get("levels")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let from_level = feeder_spec
            .get("fromLevel")
            .and_then(Value::as_str)
            .unwrap();
        let to_level = feeder_spec.get("toLevel").and_then(Value::as_str).unwrap();
        let to_inst = feeder_spec.get("toInst").and_then(Value::as_str).unwrap();

        // Build a FRESH feeder tensor and write the raw per-instance values in the
        // vector's SCRAMBLED order — `reduce_median` sorts before indexing, so the
        // result must be independent of this write order.
        let mut t = Tensor::new(levels.iter().map(String::as_str));
        for write in feeder_spec.get("writes").and_then(Value::as_array).unwrap() {
            let inst = write.get("inst").and_then(Value::as_str).unwrap();
            let option = write.get("option").and_then(Value::as_str).unwrap();
            let dim = write.get("dim").and_then(Value::as_str).unwrap();
            let value = parse_value(write.get("value").unwrap());
            let mut write_scope = Scope::new();
            if !to_inst.is_empty() {
                write_scope.insert(to_level.to_owned(), to_inst.to_owned());
            }
            write_scope.extend(scope([(from_level, inst)]));
            t.writer(from_level)
                .unwrap()
                .set_value(from_level, &write_scope, option, dim, value)
                .unwrap_or_else(|e| panic!("{file}: set_value failed: {e:?}"));
        }

        let target_scope = if to_inst.is_empty() {
            Scope::new()
        } else {
            scope([(to_level, to_inst)])
        };
        t.reduce_all_median(from_level, to_level, &target_scope)
            .unwrap_or_else(|e| panic!("{file}: reduce_all_median failed: {e:?}"));

        let mut cursor = Cursor::new();
        if let Some(obj) = vector.get("cursor").and_then(Value::as_object) {
            for (k, val) in obj {
                cursor.insert(k.clone(), val.as_str().unwrap().to_string());
            }
        }

        for want in vector.get("expected").and_then(Value::as_array).unwrap() {
            let option = want.get("option").and_then(Value::as_str).unwrap();
            let dim = want.get("dim").and_then(Value::as_str).unwrap();
            let value = want.get("value").and_then(Value::as_f64).unwrap();
            let got = t.value(&cursor, option, dim);
            assert_eq!(
                got,
                Some(value),
                "{file}: rolled-up median of ({option}, {dim}) changed. expected {value}, got {got:?}\n  {}",
                vector.get("description").and_then(Value::as_str).unwrap_or("")
            );
        }
    }
}
