// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! REGRESSION GUARD — golden `resolve` vectors, shared with the TS core.
//!
//! Each vector in `../../tests/vectors/*.json` pins one current routing decision:
//! a `{levels, cells}` tensor snapshot, a read `cursor`, and the `expected`
//! resolved order. This test rebuilds the tensor via the persistence-port wire
//! (`Tensor::from_json`), resolves at the cursor, and asserts the order matches.
//!
//! The SAME files are read by `packages/sporewright/src/golden-vectors.test.ts`,
//! making them a cross-core decision-equivalence guard (the CONTRIBUTING.md
//! "equivalence invariant"). A refactor that silently changes a routing decision
//! on EITHER core breaks here.
//!
//! These goldens were captured from the live `resolve` — never hand-fabricated.

use serde_json::Value;
use sporewright::tensor::{Cursor, Tensor};
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/vectors dir must exist") {
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
    assert!(!out.is_empty(), "no golden vectors found in {dir:?}");
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

#[test]
fn golden_resolve_vectors_pin_the_routing_decision() {
    for (file, vector) in load_vectors() {
        let tensor_json = vector
            .get("tensor")
            .unwrap_or_else(|| panic!("{file}: missing `tensor`"))
            .to_string();
        let t = Tensor::from_json(&tensor_json)
            .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));
        let cursor = cursor_from(&vector);
        let expected = expected_from(&vector);

        let resolved = t.resolve(&cursor);
        assert_eq!(
            resolved, expected,
            "{file}: resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
            vector.get("description").and_then(Value::as_str).unwrap_or("")
        );
    }
}
