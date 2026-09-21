// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden `corroborate` vectors, shared with the TS core.
//!
//! Each vector in `../../tests/corroborate-vectors/*.json` pins one decision of the
//! faker-resistant self-excluded-quorum count (named in the CONTRIBUTING equivalence
//! invariant). Before this corpus, `corroborate` was pinned only by independent
//! per-core unit tests — a one-sided change to the self-exclusion comparison
//! (`inst != option`) or the `> 0` attestation threshold would pass both suites while
//! diverging the corroboration decision. These are the first SHARED vectors both
//! cores read.
//!
//! Each vector has a `feeder` spec: the tensor `levels`, a `fromLevel`/`toLevel`/
//! `toInst`, a `quorum`, and a list of raw per-instance attestation `writes`
//! `{inst, option, dim, value}`. The runner builds a fresh feeder tensor, writes the
//! attestations, runs `corroborate`, then asserts that EVERY `materialized`
//! `{option, dim, value}` reads back at the cursor AND every `absent` `{option, dim}`
//! reads back as `None` — the exact materialized SET.
//!
//! The SAME files are read by
//! `packages/sporewright/src/corroborate-vectors.test.ts`, making them a cross-core
//! decision-equivalence guard: both cores count independent reporters with the
//! identical self-exclusion and quorum boundary and MUST materialize the identical
//! subjects. The goldens were captured from the live `corroborate` — never
//! hand-fabricated.

use serde_json::Value;
use sporewright::{scope, Cursor, Scope, Tensor};
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/corroborate-vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("corroborate-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/corroborate-vectors dir must exist") {
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
    assert!(!out.is_empty(), "no corroborate vectors found in {dir:?}");
    out
}

#[test]
fn golden_corroborate_vectors_pin_the_quorum_decision() {
    for (file, vector) in load_vectors() {
        assert_eq!(
            vector.get("kind").and_then(Value::as_str),
            Some("corroborate"),
            "{file}: only the `corroborate` vector kind is implemented"
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
        let quorum = feeder_spec.get("quorum").and_then(Value::as_u64).unwrap() as usize;

        let mut t = Tensor::new(levels.iter().map(String::as_str));
        for write in feeder_spec.get("writes").and_then(Value::as_array).unwrap() {
            let inst = write.get("inst").and_then(Value::as_str).unwrap();
            let option = write.get("option").and_then(Value::as_str).unwrap();
            let dim = write.get("dim").and_then(Value::as_str).unwrap();
            let value = write.get("value").and_then(Value::as_f64).unwrap();
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
        t.corroborate_all(from_level, to_level, &target_scope, from_level, quorum)
            .unwrap_or_else(|e| panic!("{file}: corroborate_all failed: {e:?}"));

        let mut cursor = Cursor::new();
        if let Some(obj) = vector.get("cursor").and_then(Value::as_object) {
            for (k, val) in obj {
                cursor.insert(k.clone(), val.as_str().unwrap().to_string());
            }
        }

        for want in vector
            .get("materialized")
            .and_then(Value::as_array)
            .unwrap()
        {
            let option = want.get("option").and_then(Value::as_str).unwrap();
            let dim = want.get("dim").and_then(Value::as_str).unwrap();
            let value = want.get("value").and_then(Value::as_f64).unwrap();
            let got = t.value(&cursor, option, dim);
            assert_eq!(
                got,
                Some(value),
                "{file}: subject ({option}, {dim}) should be corroborated to {value}, got {got:?}\n  {}",
                vector.get("description").and_then(Value::as_str).unwrap_or("")
            );
        }

        for gone in vector.get("absent").and_then(Value::as_array).unwrap() {
            let option = gone.get("option").and_then(Value::as_str).unwrap();
            let dim = gone.get("dim").and_then(Value::as_str).unwrap();
            let got = t.value(&cursor, option, dim);
            assert_eq!(
                got, None,
                "{file}: subject ({option}, {dim}) must NOT be corroborated (below quorum / self-claim), got {got:?}\n  {}",
                vector.get("description").and_then(Value::as_str).unwrap_or("")
            );
        }
    }
}
