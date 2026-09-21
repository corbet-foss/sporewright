// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden tensor-config vectors, shared with the TS core.
//!
//! Each vector in `../../tests/config-vectors/*.json` pins one declarative
//! seeding: a `config` (`{levels, seeds}`, where each seed is a reified `Writer`
//! call — `{floor, level, inst, option, dim, kind, value}`) and the `expected`
//! **instantiated tensor wire** (`{levels, cells}`, the `Tensor::to_json` shape).
//!
//! This test parses the config, runs [`sporewright::instantiate`], serializes the
//! result with `Tensor::to_json`, and asserts it equals the golden `expected`
//! (compared as canonical JSON values, so key order is irrelevant).
//!
//! The SAME files are read by
//! `packages/sporewright/src/config-vectors.test.ts`, making them a cross-core
//! decision-equivalence guard for the declarative builder: both cores instantiate
//! the identical config and MUST produce the byte-identical tensor — which is the
//! faithfulness claim (`a config == those imperative Writer calls`) proven on both
//! cores at once. The values come straight from the live `instantiate`/`to_json`,
//! never hand-fabricated.
//!
//! The `value` of a seed uses the same wire codec as a cell's `v`: a finite number,
//! the `"inf"`/`"-inf"`/`"nan"` sentinels for a gate, or a `{ "b64": "…" }` document.

use serde_json::{json, Value};
use sporewright::config::{instantiate, SeedCell, SeedKind, TensorConfig};
use sporewright::{Scope, Value as CellValue, WriteError};
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("config-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/config-vectors dir must exist") {
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
    assert!(!out.is_empty(), "no config vectors found in {dir:?}");
    out
}

/// Decode the seed `value` wire (same sentinels/`{b64}` shape as a cell's `v`) into a
/// [`CellValue`]. Bytes are decoded by round-tripping through the tensor's own
/// `apply_json`, but here a tiny base64 decode via the public wire is simpler: we let
/// `serde_json` parse the number/sentinel and handle `{b64}` explicitly.
fn decode_value(v: &Value) -> CellValue {
    match v {
        Value::Number(n) => CellValue::F64(n.as_f64().unwrap()),
        Value::String(s) => match s.as_str() {
            "inf" => CellValue::F64(f64::INFINITY),
            "-inf" => CellValue::F64(f64::NEG_INFINITY),
            "nan" => CellValue::F64(f64::NAN),
            other => panic!("bad value sentinel: {other}"),
        },
        Value::Object(map) => {
            let b64 = map
                .get("b64")
                .and_then(Value::as_str)
                .expect("bytes seed needs a `b64` string");
            use base64::{engine::general_purpose::STANDARD, Engine as _};
            CellValue::Bytes(STANDARD.decode(b64).expect("valid base64"))
        }
        other => panic!("bad seed value: {other}"),
    }
}

fn parse_kind(s: &str) -> SeedKind {
    match s {
        "value" => SeedKind::Value,
        "bytes" => SeedKind::Bytes,
        "weight" => SeedKind::Weight,
        "option-weight" => SeedKind::OptionWeight,
        other => panic!("unknown seed kind: {other}"),
    }
}

fn parse_config(v: &Value) -> TensorConfig {
    let layers: Vec<String> = v
        .get("layers")
        .and_then(Value::as_array)
        .expect("config.levels must be an array")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let mut cfg = TensorConfig::new(layers);
    for s in v
        .get("seeds")
        .and_then(Value::as_array)
        .expect("config.seeds must be an array")
    {
        let get = |k: &str| s.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let scope: Scope = s
            .get("scope")
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(|object| object.iter())
            .map(|(axis, value)| (axis.clone(), value.as_str().unwrap().to_owned()))
            .collect();
        cfg = cfg.with_cell(SeedCell {
            floor: get("floor"),
            layer: get("layer"),
            scope,
            option: get("option"),
            dim: get("dim"),
            kind: parse_kind(&get("kind")),
            value: decode_value(s.get("value").expect("seed needs a value")),
        });
    }
    cfg
}

#[test]
fn golden_config_vectors_pin_the_instantiated_tensor() {
    for (file, vector) in load_vectors() {
        let cfg = parse_config(
            vector
                .get("config")
                .unwrap_or_else(|| panic!("{file}: missing `config`")),
        );

        // Error-path vectors: `instantiate` must REJECT with the named WriteError,
        // identically on both cores. An error rejection is a decision too — pinning it
        // shared keeps the cores from silently diverging on which configs they accept.
        if let Some(err) = vector.get("expected_error").and_then(Value::as_str) {
            let want = match err {
                "write-up" => WriteError::WriteUp,
                "unknown-layer" => WriteError::UnknownLayer,
                "bad-value-kind" => WriteError::BadValueKind,
                other => panic!("{file}: unknown expected_error `{other}`"),
            };
            assert_eq!(
                instantiate(&cfg),
                Err(want),
                "{file}: instantiate must reject with `{err}`.\n  {}",
                vector
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
            );
            continue;
        }

        let t = instantiate(&cfg).unwrap_or_else(|e| panic!("{file}: instantiate failed: {e:?}"));

        // The instantiated tensor's wire form (canonical) must equal the golden.
        let produced: Value = serde_json::from_str(&t.to_json()).unwrap();
        let expected = vector
            .get("expected")
            .unwrap_or_else(|| panic!("{file}: missing `expected`"));
        assert_eq!(
            &produced,
            expected,
            "{file}: instantiated tensor changed.\n  {}",
            vector
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
        );

        // Belt-and-braces: if the vector also pins a resolve order, check it.
        if let Some(exp) = vector.get("resolve_expected").and_then(Value::as_array) {
            let cursor = {
                let mut c = sporewright::Cursor::new();
                if let Some(obj) = vector.get("cursor").and_then(Value::as_object) {
                    for (k, val) in obj {
                        c.insert(k.clone(), val.as_str().unwrap().to_string());
                    }
                }
                c
            };
            let resolved = t.resolve(&cursor);
            let want: Vec<String> = exp
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            assert_eq!(resolved, want, "{file}: resolve order changed");
        }

        let _ = json!({}); // keep serde_json::json import used regardless of branch.
    }
}
