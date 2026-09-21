// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! CROSS-CORE GUARD — golden budget/price + token-bucket vectors, shared with the
//! TS core.
//!
//! Each vector in `../../tests/budget-vectors/*.json` pins one budget-layer decision.
//! Three kinds:
//!
//! * `kind="price"`: a base `{levels, cells}` tensor, a `budget` spec (the constant
//!   step, the price write level, the pools `{cap}`, and the option→pool mapping), and
//!   a sequence of `ticks` (each a list of `{pool, usage}`). After every tick the
//!   prices are published as the **shared** `budget:<pool>` weight and the tensor is
//!   resolved. Asserts the `expectedOrder` and the final `expectedLambda` per pool. The
//!   `ticks` feed `measured` DIRECTLY, bypassing the raw-usage → `reduce_sum` →
//!   `aggregate_usage` feeder.
//! * `kind="usage"`: exercises the FEEDER path the `price` kind bypasses. A `feeder`
//!   spec lists raw per-instance usage writes `{option, inst, usage}` in a SPECIFIC
//!   order (insertion order is load-bearing — it must differ from the canonical sorted
//!   order so the sum-order fix is actually under test). The runner builds a fresh
//!   feeder tensor, writes the usage in the given order, runs `aggregate_usage` (which
//!   internally calls `reduce_sum` per option then sums per-option totals per pool),
//!   then `tick` → publish → resolve on the base tensor. Asserts `expectedMeasured`
//!   (the exact per-pool sums — a bit-for-bit cross-core check), `expectedLambda`, and
//!   `expectedOrder`. This is the only kind that guards `reduce_sum`'s instance-sum
//!   order AND `aggregate_usage`'s option-sum order — both diverge across cores unless
//!   summed canonically.
//! * `kind="bucket"`: a `bucket` spec (`capacity`, `refillPerSec`, `start`) and a list
//!   of `admits` times; asserts the `expectedAdmits` booleans.
//!
//! The SAME files are read by `packages/sporewright/src/budget-vectors.test.ts`,
//! making them a cross-core decision-equivalence guard: both cores build the identical
//! `Budget`/`TokenBucket`, drive the identical step sequence, and MUST reach the
//! identical resolved order, final prices, and admit decisions. The goldens were
//! captured from the live budget layer — never hand-fabricated.

use serde_json::Value;
use sporewright::{scope, Budget, Cursor, Scope, Tensor, TokenBucket};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/sporewright; vectors live at repo-root tests/budget-vectors.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("budget-vectors")
}

fn load_vectors() -> Vec<(String, Value)> {
    let dir = vectors_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests/budget-vectors dir must exist") {
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
    assert!(!out.is_empty(), "no budget vectors found in {dir:?}");
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

fn build_budget(spec: &Value) -> (Budget, String) {
    let step = spec.get("step").and_then(Value::as_f64).unwrap();
    let level = spec
        .get("priceLevel")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    let mut b = Budget::new().with_step(step);
    for p in spec.get("pools").and_then(Value::as_array).unwrap() {
        let pool = p.get("pool").and_then(Value::as_str).unwrap();
        let cap = p.get("cap").and_then(Value::as_f64).unwrap();
        b = b.with_pool(pool, cap);
    }
    for o in spec.get("options").and_then(Value::as_array).unwrap() {
        let option = o.get("option").and_then(Value::as_str).unwrap();
        let pool = o.get("pool").and_then(Value::as_str).unwrap();
        b = b.map_option(option, pool);
    }
    (b, level)
}

fn run_price_vector(file: &str, vector: &Value) {
    let tensor_json = vector
        .get("tensor")
        .unwrap_or_else(|| panic!("{file}: missing `tensor`"))
        .to_string();
    let mut t = Tensor::from_json(&tensor_json)
        .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));

    let spec = vector
        .get("budget")
        .unwrap_or_else(|| panic!("{file}: missing `budget`"));
    let (mut budget, level) = build_budget(spec);

    let ticks = vector.get("ticks").and_then(Value::as_array).unwrap();
    for tick in ticks {
        let mut measured: BTreeMap<String, f64> = BTreeMap::new();
        for entry in tick.as_array().unwrap() {
            let pool = entry.get("pool").and_then(Value::as_str).unwrap();
            let usage = entry.get("usage").and_then(Value::as_f64).unwrap();
            measured.insert(pool.to_string(), usage);
        }
        budget.tick(&measured);
        let mut w = t.writer(&level).unwrap();
        budget
            .publish_prices(&mut w, &level, &Scope::new())
            .unwrap_or_else(|e| panic!("{file}: publish_prices failed: {e:?}"));
    }
    // If there were no ticks at all, still publish the (zero) prices once.
    if ticks.is_empty() {
        let mut w = t.writer(&level).unwrap();
        budget
            .publish_prices(&mut w, &level, &Scope::new())
            .unwrap();
    }

    let cursor = cursor_from(vector);
    let expected: Vec<String> = vector
        .get("expectedOrder")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let resolved = t.resolve(&cursor);
    assert_eq!(
        resolved, expected,
        "{file}: budget-priced resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
        vector.get("description").and_then(Value::as_str).unwrap_or("")
    );

    if let Some(lambdas) = vector.get("expectedLambda").and_then(Value::as_object) {
        for (pool, lam) in lambdas {
            let want = lam.as_f64().unwrap();
            let got = budget.lambda(pool);
            assert_eq!(
                got, want,
                "{file}: pool `{pool}` price λ changed. expected {want}, got {got}"
            );
        }
    }
}

fn run_usage_vector(file: &str, vector: &Value) {
    let tensor_json = vector
        .get("tensor")
        .unwrap_or_else(|| panic!("{file}: missing `tensor`"))
        .to_string();
    let mut t = Tensor::from_json(&tensor_json)
        .unwrap_or_else(|| panic!("{file}: `tensor` did not rebuild via from_json"));

    let spec = vector
        .get("budget")
        .unwrap_or_else(|| panic!("{file}: missing `budget`"));
    let (mut budget, level) = build_budget(spec);

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

    // Build a FRESH feeder tensor and write the raw per-instance usage in the vector's
    // INSERTION order (load-bearing: it must differ from the canonical sorted order so
    // the sum-order fix is under test).
    let mut feeder = Tensor::new(levels.iter().map(String::as_str));
    for write in feeder_spec.get("writes").and_then(Value::as_array).unwrap() {
        let option = write.get("option").and_then(Value::as_str).unwrap();
        let inst = write.get("inst").and_then(Value::as_str).unwrap();
        let usage = write.get("usage").and_then(Value::as_f64).unwrap();
        let mut write_scope = Scope::new();
        if !to_inst.is_empty() {
            write_scope.insert(to_level.to_owned(), to_inst.to_owned());
        }
        write_scope.extend(scope([(from_level, inst)]));
        let mut w = feeder.writer(from_level).unwrap();
        budget
            .write_usage(&mut w, from_level, &write_scope, option, usage)
            .unwrap_or_else(|e| panic!("{file}: write_usage failed: {e:?}"));
    }

    // aggregate_usage internally reduce_sums per option then sums per-option totals per
    // pool — both summations must be canonical for cross-core equivalence.
    let target_scope = if to_inst.is_empty() {
        Scope::new()
    } else {
        scope([(to_level, to_inst)])
    };
    let measured = budget.aggregate_usage(&mut feeder, from_level, to_level, &target_scope);
    // Exact (bit-for-bit) per-pool measured sums — the primary divergence guard.
    if let Some(want) = vector.get("expectedMeasured").and_then(Value::as_object) {
        for (pool, v) in want {
            let want_v = v.as_f64().unwrap();
            let got = measured.get(pool).copied().unwrap_or(0.0);
            assert_eq!(
                got, want_v,
                "{file}: pool `{pool}` measured sum changed. expected {want_v}, got {got}"
            );
        }
    }

    budget.tick(&measured);
    let mut w = t.writer(&level).unwrap();
    budget
        .publish_prices(&mut w, &level, &Scope::new())
        .unwrap_or_else(|e| panic!("{file}: publish_prices failed: {e:?}"));

    if let Some(lambdas) = vector.get("expectedLambda").and_then(Value::as_object) {
        for (pool, lam) in lambdas {
            let want = lam.as_f64().unwrap();
            let got = budget.lambda(pool);
            assert_eq!(
                got, want,
                "{file}: pool `{pool}` price λ changed. expected {want}, got {got}"
            );
        }
    }

    let cursor = cursor_from(vector);
    let expected: Vec<String> = vector
        .get("expectedOrder")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let resolved = t.resolve(&cursor);
    assert_eq!(
        resolved, expected,
        "{file}: usage-fed resolve order changed.\n  expected (golden): {expected:?}\n  actual   (now):    {resolved:?}\n  {}",
        vector.get("description").and_then(Value::as_str).unwrap_or("")
    );
}

fn run_bucket_vector(file: &str, vector: &Value) {
    let spec = vector
        .get("bucket")
        .unwrap_or_else(|| panic!("{file}: missing `bucket`"));
    let capacity = spec.get("capacity").and_then(Value::as_f64).unwrap();
    let refill = spec.get("refillPerSec").and_then(Value::as_f64).unwrap();
    let start = spec.get("start").and_then(Value::as_f64).unwrap();
    let mut tb = TokenBucket::new(capacity, refill, start);

    let admits = vector.get("admits").and_then(Value::as_array).unwrap();
    let expected: Vec<bool> = vector
        .get("expectedAdmits")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect();
    let got: Vec<bool> = admits
        .iter()
        .map(|now| tb.admit(now.as_f64().unwrap()))
        .collect();
    assert_eq!(
        got, expected,
        "{file}: token-bucket admit decisions changed.\n  expected (golden): {expected:?}\n  actual   (now):    {got:?}\n  {}",
        vector.get("description").and_then(Value::as_str).unwrap_or("")
    );
}

#[test]
fn golden_budget_vectors_pin_the_budget_decision() {
    for (file, vector) in load_vectors() {
        match vector.get("kind").and_then(Value::as_str) {
            Some("price") => run_price_vector(&file, &vector),
            Some("usage") => run_usage_vector(&file, &vector),
            Some("bucket") => run_bucket_vector(&file, &vector),
            other => panic!("{file}: unknown vector kind {other:?}"),
        }
    }
}
