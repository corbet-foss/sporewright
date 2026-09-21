// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Shared trust vectors: the same JSON is executed by the Rust and TypeScript
//! cores so product-side persistence bindings cannot observe different scores.

use serde::Deserialize;
use sporewright::update_pair_trust;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Vector {
    name: String,
    incoming_trust: f64,
    current_trust: f64,
    agree: bool,
    freshness: f64,
    source_volatility: f64,
    expected_incoming_trust: f64,
    expected_current_trust: f64,
}

#[test]
fn shared_trust_vectors() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/trust-vectors");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .expect("tests/trust-vectors must exist")
        .map(|entry| entry.expect("vector entry").path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no trust vectors in {dir:?}");

    for path in paths {
        let vector: Vector = serde_json::from_str(
            &fs::read_to_string(&path).expect("trust vector must be readable"),
        )
        .expect("trust vector must be valid JSON");
        let result = update_pair_trust(
            vector.incoming_trust,
            vector.current_trust,
            vector.agree,
            vector.freshness,
            vector.source_volatility,
        );
        assert!(
            (result.incoming_trust - vector.expected_incoming_trust).abs() < 1e-12,
            "{} incoming score diverged",
            vector.name
        );
        assert!(
            (result.current_trust - vector.expected_current_trust).abs() < 1e-12,
            "{} current score diverged",
            vector.name
        );
    }
}
