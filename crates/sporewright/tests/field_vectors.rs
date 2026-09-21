// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Shared addressed-field vectors, executed by both public cores.

use serde::Deserialize;
use sporewright::field::{Address, AddressedField, DecisionPolicy};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Declaration {
    kind: String,
    address: Address,
    option: String,
    dimension: String,
    value: f64,
}

#[derive(Deserialize)]
struct Observation {
    address: Address,
    option: String,
    dimension: String,
    value: f64,
    variance: f64,
}

#[derive(Deserialize)]
struct Maintenance {
    kind: String,
    address: Address,
    option: String,
    dimension: String,
    factor: Option<f64>,
}

#[derive(Deserialize)]
struct Query {
    address: Address,
    temperature: f64,
    #[serde(default)]
    allowed: Option<Vec<String>>,
    expected: Vec<String>,
    means: BTreeMap<String, BTreeMap<String, f64>>,
}

#[derive(Deserialize)]
struct Vector {
    name: String,
    layers: Vec<String>,
    declarations: Vec<Declaration>,
    observations: Vec<Observation>,
    #[serde(default)]
    maintenance: Vec<Maintenance>,
    queries: Vec<Query>,
}

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("field-vectors")
}

#[test]
fn addressed_field_vectors_are_decision_equivalent() {
    let mut paths: Vec<_> = fs::read_dir(vectors_dir())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    for path in paths {
        let vector: Vector = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let mut field = AddressedField::new(vector.layers.clone());
        {
            let mut writer = field.root_writer();
            for declaration in vector.declarations {
                match declaration.kind.as_str() {
                    "prior" => writer
                        .set_prior(
                            &declaration.address,
                            &declaration.option,
                            &declaration.dimension,
                            declaration.value,
                        )
                        .unwrap(),
                    "weight" => writer
                        .set_option_weight(
                            &declaration.address,
                            &declaration.option,
                            &declaration.dimension,
                            declaration.value,
                        )
                        .unwrap(),
                    "gate" => writer
                        .set_gate(
                            &declaration.address,
                            &declaration.option,
                            &declaration.dimension,
                            declaration.value > 0.0,
                        )
                        .unwrap(),
                    other => panic!("{}: unknown declaration kind {other}", vector.name),
                }
            }
            for observation in vector.observations {
                writer
                    .observe(
                        &observation.address,
                        &observation.option,
                        &observation.dimension,
                        observation.value,
                        observation.variance,
                    )
                    .unwrap();
            }
            for maintenance in vector.maintenance {
                match maintenance.kind.as_str() {
                    "compact" => {
                        writer
                            .compact_subtree(
                                &maintenance.address,
                                &maintenance.option,
                                &maintenance.dimension,
                            )
                            .unwrap();
                    }
                    "discount" => {
                        writer
                            .discount_evidence(
                                &maintenance.address,
                                &maintenance.option,
                                &maintenance.dimension,
                                maintenance.factor.unwrap(),
                            )
                            .unwrap();
                    }
                    other => panic!("{}: unknown maintenance kind {other}", vector.name),
                }
            }
        }
        for query in vector.queries {
            let policy = DecisionPolicy {
                temperature: query.temperature,
            };
            let decision = match query.allowed {
                Some(allowed) => field.decide_among(&query.address, &allowed, policy),
                None => field.decide(&query.address, policy),
            }
            .unwrap();
            let resolved: Vec<_> = decision
                .alternatives
                .iter()
                .filter(|candidate| candidate.viable)
                .map(|candidate| candidate.option.clone())
                .collect();
            assert_eq!(resolved, query.expected, "{}: route order", vector.name);
            for (option, dimensions) in query.means {
                let candidate = decision
                    .alternatives
                    .iter()
                    .find(|candidate| candidate.option == option)
                    .unwrap();
                for (dimension, expected) in dimensions {
                    let actual = candidate
                        .dimensions
                        .iter()
                        .find(|trace| trace.dimension == dimension)
                        .unwrap()
                        .mean;
                    assert!(
                        (actual - expected).abs() < 1e-10,
                        "{}: {option}/{dimension}: {actual} != {expected}",
                        vector.name
                    );
                }
            }
        }
    }
}
