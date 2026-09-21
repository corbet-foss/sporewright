// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden capability-gate vectors, shared with the TS core.
//!
//! Each vector in `../../tests/gate-vectors/*.json` pins one gating decision: a
//! `{levels, cells}` base tensor, a `gate` spec (writer level, the slice
//! `(level, inst)`, the option ids, the required caps, and a serialized
//! `offers` map standing in for the predicate), a read `cursor`, and the
//! `expected` resolved order **after** the gate is applied.
//!
//! This test rebuilds the tensor via the persistence-port wire
//! (`Tensor::from_json`), runs [`sporewright::gate_capabilities`] with an offers
//! lookup built from the map, resolves at the cursor, and asserts the order.
//!
//! The SAME files are read by
//! `packages/sporewright/src/gate-vectors.test.ts`, making them a cross-core
//! decision-equivalence guard for the gate primitive: both cores apply their own
//! `gate_capabilities` to the identical base tensor and MUST drop the identical
//! options and return the identical survivor order.

use serde_json::Value;
use sporewright::{gate_capabilities, Cursor, Tensor};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/gate-vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("gate-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/gate-vectors dir must exist") {
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
    assert!(!out.is_empty(), "no gate vectors found in {dir:?}");
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

fn str_array(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("missing `{key}` array"))
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn golden_gate_vectors_pin_the_gating_decision() {
    for (file, vector) in load_vectors() {
        let tensor_json = vector
            .get("tensor")
            .unwrap_or_else(|| panic!("{file}: missing `tensor`"))
            .to_string();
        let mut t = Tensor::from_json(&tensor_json)
            .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));

        let gate = vector
            .get("gate")
            .unwrap_or_else(|| panic!("{file}: missing `gate`"));
        let writer_level = gate.get("writerLevel").and_then(Value::as_str).unwrap();
        let level = gate.get("level").and_then(Value::as_str).unwrap();
        let inst = gate.get("inst").and_then(Value::as_str).unwrap();
        let options = str_array(gate, "options");
        let required = str_array(gate, "required");

        // The serialized offers map: option -> the caps it provides.
        let mut offers_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(obj) = gate.get("offers").and_then(Value::as_object) {
            for (opt, caps) in obj {
                let caps = caps
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap().to_string())
                    .collect();
                offers_map.insert(opt.clone(), caps);
            }
        }
        let offers = |option: &str, cap: &str| -> bool {
            offers_map
                .get(option)
                .is_some_and(|caps| caps.iter().any(|c| c == cap))
        };

        {
            let mut w = t.writer(writer_level).unwrap();
            let opt_refs: Vec<&str> = options.iter().map(String::as_str).collect();
            let req_refs: Vec<&str> = required.iter().map(String::as_str).collect();
            gate_capabilities(&mut w, level, inst, &opt_refs, &req_refs, offers)
                .unwrap_or_else(|e| panic!("{file}: gate_capabilities failed: {e:?}"));
        }

        let cursor = cursor_from(&vector);
        let expected = str_array(&vector, "expected");
        let resolved = t.resolve(&cursor);
        assert_eq!(
            resolved, expected,
            "{file}: gated resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
            vector.get("description").and_then(Value::as_str).unwrap_or("")
        );
    }
}
