// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! The one sparse tensor of facts — the heart of the wright. See `docs/MODEL.md`.
//!
//! A **cell** is a fact at a coordinate `(level, instance, option, dimension)`,
//! holding two co-located layers: a **value** (the learned truth) and a **weight**
//! (the governing will). Both are sparse. Reading is a **cursor** (one instance per
//! context tier); the fold takes the most-specific set layer. `resolve` is
//! `Σ value·weight`.
//!
//! **No clock, no CRDT.** Versioning, durability, delta, dedup, and deletion are a
//! persistence-port concern, satisfied by the product's database — the tensor core
//! is a clock-free *lens*. `to_json`/`apply_json` serialize cells for the port (and
//! for shipping a snapshot to a client); the *version* a client tracks is the DB's,
//! not the tensor's.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

fn fcmp(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(match (a.is_nan(), b.is_nan()) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => Ordering::Equal,
    })
}

/// The shared slot — the curled `""` (an unset instance, or the option-shared
/// weight that applies to every contender).
pub const SHARED: &str = "";

/// A read cursor: one instance per context tier; an absent tier means `""`.
pub type Cursor = BTreeMap<String, String>;

/// A cell value: a routable scalar (`F64`) or an opaque document blob (`Bytes`),
/// inert to the routing math.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    F64(f64),
    Bytes(Vec<u8>),
}

impl Value {
    /// The routable scalar, or `None` for a document cell.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(x) => Some(*x),
            Value::Bytes(_) => None,
        }
    }
    /// The document bytes, or `None` for a scalar cell.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            Value::F64(_) => None,
        }
    }
}

/// A fact at a coordinate: the two co-located layers, each optional and sparse.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Cell {
    value: Option<Value>,
    weight: Option<Value>,
}

/// Rejected because it would write *up* the hierarchy, name an unknown tier, or carry a
/// payload whose declared kind disagrees with its value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// The write targets a tier coarser than the writer's floor (write-up).
    WriteUp,
    /// The named floor or level tier does not exist in the skeleton.
    UnknownLevel,
    /// The seed's declared kind disagrees with its value payload (e.g. a `Bytes`/scalar
    /// mismatch) — distinct from an unknown tier so the author is pointed at the
    /// malformed cell, not the levels list.
    BadValueKind,
}

/// Wire encoding for a value: finite scalar → a readable JSON number; the
/// non-finite `±∞`/`NaN` gates → sentinel strings; a document → a `{b64}` object.
mod wire_value {
    use super::Value;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Value, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Value::F64(x) => {
                let x = *x;
                if x.is_finite() {
                    s.serialize_f64(x)
                } else if x.is_nan() {
                    s.serialize_str("nan")
                } else if x > 0.0 {
                    s.serialize_str("inf")
                } else {
                    s.serialize_str("-inf")
                }
            }
            Value::Bytes(b) => {
                use serde::ser::SerializeMap;
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("b64", &STANDARD.encode(b))?;
                m.end()
            }
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Num(f64),
        Tag(String),
        Bytes { b64: String },
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        match Repr::deserialize(d)? {
            Repr::Num(n) => Ok(Value::F64(n)),
            Repr::Tag(t) => match t.as_str() {
                "inf" => Ok(Value::F64(f64::INFINITY)),
                "-inf" => Ok(Value::F64(f64::NEG_INFINITY)),
                "nan" => Ok(Value::F64(f64::NAN)),
                other => Err(D::Error::custom(format!("bad value sentinel: {other}"))),
            },
            Repr::Bytes { b64 } => STANDARD
                .decode(b64.as_bytes())
                .map(Value::Bytes)
                .map_err(D::Error::custom),
        }
    }
}

/// `Option<Value>` wire codec reusing [`wire_value`].
mod opt_value {
    use super::{wire_value, Value};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    struct Wrap<'a>(&'a Value);
    impl Serialize for Wrap<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            wire_value::serialize(self.0, s)
        }
    }

    pub fn serialize<S: Serializer>(v: &Option<Value>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(x) => s.serialize_some(&Wrap(x)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
        #[derive(Deserialize)]
        struct W(#[serde(with = "wire_value")] Value);
        Ok(Option::<W>::deserialize(d)?.map(|w| w.0))
    }
}

/// One serialized cell — the unit the persistence port stores and ships. No clock:
/// the DB versions the row, the cell carries only its coordinate and layers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonCell {
    pub level: String,
    pub inst: String,
    pub option: String,
    pub dim: String,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_value")]
    pub v: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_value")]
    pub w: Option<Value>,
}

fn cell_order(a: &JsonCell, b: &JsonCell) -> Ordering {
    (&a.level, &a.inst, &a.option, &a.dim).cmp(&(&b.level, &b.inst, &b.option, &b.dim))
}

type Cells = BTreeMap<String, BTreeMap<String, BTreeMap<String, BTreeMap<String, Cell>>>>;

/// Which layer of a cell a write targets — internal, never serialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    Value,
    Weight,
}

/// The tensor: a sparse, clock-free lens over a fixed, ordered context-tier
/// skeleton. Durability and versioning live in the persistence port.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// The context-tier skeleton, coarsest first. Fixed at construction.
    pub levels: Vec<String>,
    cells: Cells,
}

impl Tensor {
    /// A tensor over a fixed, ordered context-tier skeleton (coarsest first).
    pub fn new(levels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Tensor {
            levels: levels.into_iter().map(Into::into).collect(),
            cells: BTreeMap::new(),
        }
    }

    fn level_index(&self, level: &str) -> Option<usize> {
        self.levels.iter().position(|l| l == level)
    }

    /// Mint a [`Writer`] that may set cells at `level` or **finer** (write-down).
    pub fn writer<'a>(&'a mut self, level: &str) -> Option<Writer<'a>> {
        let floor = self.level_index(level)?;
        Some(Writer {
            tensor: self,
            floor,
        })
    }

    fn put(
        &mut self,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        layer: Layer,
        value: Value,
    ) {
        let cell = self
            .cells
            .entry(level.into())
            .or_default()
            .entry(inst.into())
            .or_default()
            .entry(option.into())
            .or_default()
            .entry(dim.into())
            .or_default();
        match layer {
            Layer::Value => cell.value = Some(value),
            Layer::Weight => cell.weight = Some(value),
        }
    }

    /// Remove the entire `(level, inst)` slice — every option/dim cell at that
    /// coordinate. The orchestrator uses this to prune per-job gate cells from the
    /// long-lived control-plane tensor (a re-routed or completed job must not leave its
    /// `priv:*` cells behind). A no-op if the slice is absent. Mirrors the TS
    /// `clearLevelInst`.
    pub fn clear_level_inst(&mut self, level: &str, inst: &str) {
        if let Some(m) = self.cells.get_mut(level) {
            m.remove(inst);
        }
    }

    fn inst<'a>(cursor: &'a Cursor, level: &str) -> &'a str {
        cursor.get(level).map(String::as_str).unwrap_or(SHARED)
    }

    fn get(&self, level: &str, inst: &str, option: &str, dim: &str) -> Option<&Cell> {
        self.cells.get(level)?.get(inst)?.get(option)?.get(dim)
    }

    /// Fold one layer over the context tiers at `cursor`: the most-specific (deepest)
    /// tier that sets it wins.
    fn fold(&self, cursor: &Cursor, option: &str, dim: &str, layer: Layer) -> Option<&Value> {
        let mut out = None;
        for level in &self.levels {
            let v = self
                .get(level, Self::inst(cursor, level), option, dim)
                .and_then(|c| match layer {
                    Layer::Value => c.value.as_ref(),
                    Layer::Weight => c.weight.as_ref(),
                });
            if v.is_some() {
                out = v;
            }
        }
        out
    }

    /// Effective scalar **value** at `cursor` (deepest set; a document reads `None`).
    pub fn value(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<f64> {
        self.fold(cursor, option, dim, Layer::Value)
            .and_then(Value::as_f64)
    }

    /// Effective document **bytes** at `cursor` (deepest set, if a `Bytes` cell).
    pub fn bytes(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<&[u8]> {
        self.fold(cursor, option, dim, Layer::Value)
            .and_then(Value::as_bytes)
    }

    /// Effective **weight** of `(option, dim)`: option-specific over option-shared.
    pub fn weight(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<f64> {
        self.fold(cursor, option, dim, Layer::Weight)
            .or_else(|| self.fold(cursor, SHARED, dim, Layer::Weight))
            .and_then(Value::as_f64)
    }

    /// The options visible at the `cursor` (the shared slot is not an option).
    pub fn options(&self, cursor: &Cursor) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for level in &self.levels {
            if let Some(m) = self
                .cells
                .get(level)
                .and_then(|m| m.get(Self::inst(cursor, level)))
            {
                for o in m.keys() {
                    if o != SHARED {
                        out.insert(o.clone());
                    }
                }
            }
        }
        out
    }

    fn dims_of(&self, cursor: &Cursor, option: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for level in &self.levels {
            if let Some(m) = self
                .cells
                .get(level)
                .and_then(|m| m.get(Self::inst(cursor, level)))
                .and_then(|m| m.get(option))
            {
                out.extend(m.keys().cloned());
            }
        }
        out
    }

    /// **resolve** — `Σ value·weight` per option, drop a gate (a non-finite `value` —
    /// `±∞`/`NaN`), order ascending → the queue. A pure-document option is not a candidate.
    pub fn resolve(&self, cursor: &Cursor) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = Vec::new();
        'opt: for option in self.options(cursor) {
            let mut score = 0.0;
            let mut routable = false;
            for dim in self.dims_of(cursor, &option) {
                let v = match self.value(cursor, &option, &dim) {
                    None => continue,
                    Some(v) => v,
                };
                routable = true;
                if !v.is_finite() {
                    continue 'opt;
                }
                if let Some(w) = self.weight(cursor, &option, &dim) {
                    if w.is_finite() {
                        score += w * v;
                    }
                }
            }
            if routable {
                scored.push((option, score));
            }
        }
        scored.sort_by(|a, b| fcmp(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
        scored.into_iter().map(|(o, _)| o).collect()
    }

    /// **reduce** — the one value-path up: median of the value layer over every
    /// instance at `from_level` for `(option, dim)`, written to `to_level`/`to_inst`.
    pub fn reduce_median(
        &mut self,
        from_level: &str,
        to_level: &str,
        to_inst: &str,
        option: &str,
        dim: &str,
    ) {
        let mut vals: Vec<f64> = self
            .cells
            .get(from_level)
            .into_iter()
            .flat_map(|m| m.values())
            .filter_map(|m| m.get(option)?.get(dim)?.value.as_ref()?.as_f64())
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            return;
        }
        vals.sort_by(|a, b| fcmp(*a, *b));
        let n = vals.len();
        let med = if n % 2 == 1 {
            vals[n / 2]
        } else {
            (vals[n / 2 - 1] + vals[n / 2]) / 2.0
        };
        self.put(
            to_level,
            to_inst,
            option,
            dim,
            Layer::Value,
            Value::F64(med),
        );
    }

    /// **reduce_sum** — the additive value-path up: the *sum* of the value layer over
    /// every instance at `from_level` for `(option, dim)`, written to `to_level`/`to_inst`.
    /// The budget layer's feeder: total draw on a shared resource pool is a sum, not a
    /// median. Non-finite values are excluded; no support writes nothing.
    ///
    /// **The sum order is load-bearing for cross-core decision-equivalence, not cosmetic.**
    /// IEEE-754 addition is non-associative, so summing the same draws in a different
    /// order yields a last-ULP-different total — which, fed through `tick → λ → resolve`,
    /// can flip a tie and produce the OPPOSITE queue on one core. Rust iterates instances
    /// via a `BTreeMap` (sorted by name) while TS iterates a `Map` (insertion order), so
    /// the *collection* order already differs across cores (e.g. once ≥10 jobs exist the
    /// string sort `j0,j1,j10,j2,…` ≠ insertion `j0,j1,…,j10`). We therefore **sort the
    /// collected values ascending** before summing: this removes all dependence on map
    /// iteration order and is trivially identical on both cores. Same discipline as
    /// `dims_of` (the resolve sum order) and `aggregate_usage` (the pool sum order).
    pub fn reduce_sum(
        &mut self,
        from_level: &str,
        to_level: &str,
        to_inst: &str,
        option: &str,
        dim: &str,
    ) -> Option<f64> {
        let mut vals: Vec<f64> = self
            .cells
            .get(from_level)
            .into_iter()
            .flat_map(|m| m.values())
            .filter_map(|m| m.get(option)?.get(dim)?.value.as_ref()?.as_f64())
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            return None;
        }
        // Canonical (ascending) order so both cores add identical operands in identical
        // order — IEEE-754 addition is non-associative; see the doc comment above.
        vals.sort_by(|a, b| fcmp(*a, *b));
        let sum: f64 = vals.iter().sum();
        self.put(
            to_level,
            to_inst,
            option,
            dim,
            Layer::Value,
            Value::F64(sum),
        );
        Some(sum)
    }

    /// **roll up** — `reduce_median` for every scalar `(option, dim)` at `from_level`.
    pub fn roll_up(&mut self, from_level: &str, to_level: &str, to_inst: &str) {
        for (opt, dim) in self.pairs_at(from_level) {
            self.reduce_median(from_level, to_level, to_inst, &opt, &dim);
        }
    }

    /// **corroborate** — the option-axis sibling of `reduce`, rolling *existence* up
    /// instead of magnitude. For each subject `(option, dim)` attested at
    /// `from_level`, count the **independent** reporters (instance ≠ option —
    /// self-report excluded) whose value is a positive attestation (`> 0`); if that
    /// count reaches `quorum`, materialize the subject at `to_level`/`to_inst` as a
    /// corroborated `1.0`. Faker-resistant: a subject cannot vouch for itself, and
    /// (with live-scoping in the port) each reporter must be a live, distinct writer.
    pub fn corroborate(&mut self, from_level: &str, to_level: &str, to_inst: &str, quorum: usize) {
        for (opt, dim) in self.pairs_at(from_level) {
            let count = self
                .cells
                .get(from_level)
                .into_iter()
                .flat_map(|by_inst| by_inst.iter())
                .filter(|(inst, _)| inst.as_str() != opt) // self-exclusion
                .filter(|(_, by_opt)| {
                    by_opt
                        .get(&opt)
                        .and_then(|m| m.get(&dim))
                        .and_then(|c| c.value.as_ref())
                        .and_then(Value::as_f64)
                        .is_some_and(|v| v > 0.0)
                })
                .count();
            if count >= quorum {
                self.put(to_level, to_inst, &opt, &dim, Layer::Value, Value::F64(1.0));
            }
        }
    }

    /// The distinct `(option, dim)` pairs present at `from_level`, in canonical order.
    fn pairs_at(&self, from_level: &str) -> BTreeSet<(String, String)> {
        self.cells
            .get(from_level)
            .into_iter()
            .flat_map(|by_inst| by_inst.values())
            .flat_map(|by_opt| {
                by_opt.iter().flat_map(|(opt, by_dim)| {
                    by_dim.keys().map(move |dim| (opt.clone(), dim.clone()))
                })
            })
            .collect()
    }

    /// How many instances at `from_level` carry a finite value for `(option, dim)` —
    /// the evidence count the orchestrator reads to find thin cells.
    pub fn support(&self, from_level: &str, option: &str, dim: &str) -> usize {
        self.cells
            .get(from_level)
            .into_iter()
            .flat_map(|m| m.values())
            .filter(|m| {
                m.get(option)
                    .and_then(|d| d.get(dim))
                    .and_then(|c| c.value.as_ref())
                    .and_then(Value::as_f64)
                    .is_some_and(f64::is_finite)
            })
            .count()
    }

    // ---- Serialization (the persistence port's currency; no clocks) ----------

    /// Every cell as a flat, canonically-sorted list — what the port persists to the
    /// DB and ships to a client. Versioning (a `since=N` watermark) is the DB's, not
    /// the tensor's, so it is not part of this.
    pub fn cells(&self) -> Vec<JsonCell> {
        let mut out = Vec::new();
        for (lv, by_inst) in &self.cells {
            for (inst, by_opt) in by_inst {
                for (opt, by_dim) in by_opt {
                    for (dim, cell) in by_dim {
                        out.push(JsonCell {
                            level: lv.clone(),
                            inst: inst.clone(),
                            option: opt.clone(),
                            dim: dim.clone(),
                            v: cell.value.clone(),
                            w: cell.weight.clone(),
                        });
                    }
                }
            }
        }
        out.sort_by(cell_order);
        out
    }

    /// Serialize the whole tensor (`{levels, cells}`) for storage or a client ship.
    pub fn to_json(&self) -> String {
        let levels = serde_json::to_string(&self.levels).unwrap_or_else(|_| "[]".into());
        let cells = serde_json::to_string(&self.cells()).unwrap_or_else(|_| "[]".into());
        format!(r#"{{"levels":{levels},"cells":{cells}}}"#)
    }

    /// Overlay a list of cells (a snapshot or a DB delta) onto this tensor — a plain
    /// overwrite, since the DB already decided which row is current. Malformed cells
    /// are skipped.
    pub fn apply_json(&mut self, cells_json: &str) {
        let raw: Vec<serde_json::Value> = serde_json::from_str(cells_json).unwrap_or_default();
        for value in raw {
            let c: JsonCell = match serde_json::from_value(value) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if let Some(v) = c.v {
                self.put(&c.level, &c.inst, &c.option, &c.dim, Layer::Value, v);
            }
            if let Some(w) = c.w {
                self.put(&c.level, &c.inst, &c.option, &c.dim, Layer::Weight, w);
            }
        }
    }

    /// Rebuild a tensor from [`Tensor::to_json`].
    pub fn from_json(s: &str) -> Option<Tensor> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let levels: Vec<String> = serde_json::from_value(v.get("levels")?.clone()).ok()?;
        let mut t = Tensor::new(levels);
        let cells = serde_json::to_string(v.get("cells")?).ok()?;
        t.apply_json(&cells);
        Some(t)
    }
}

/// A capability handle: writes at its floor tier or **finer**, never coarser.
/// (No clock — the DB versions the write; this enforces only write-*down*.)
pub struct Writer<'a> {
    tensor: &'a mut Tensor,
    floor: usize,
}

impl Writer<'_> {
    fn check(&self, level: &str) -> Result<(), WriteError> {
        match self.tensor.level_index(level) {
            None => Err(WriteError::UnknownLevel),
            Some(i) if i < self.floor => Err(WriteError::WriteUp),
            Some(_) => Ok(()),
        }
    }

    fn write(
        &mut self,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        layer: Layer,
        value: Value,
    ) -> Result<(), WriteError> {
        self.check(level)?;
        self.tensor.put(level, inst, option, dim, layer, value);
        Ok(())
    }

    /// Set a scalar **value** at `(level, instance, option, dimension)`.
    pub fn set_value(
        &mut self,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        v: f64,
    ) -> Result<(), WriteError> {
        self.write(level, inst, option, dim, Layer::Value, Value::F64(v))
    }

    /// Set a **document** blob at `(level, instance, option, dimension)`.
    pub fn set_bytes(
        &mut self,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        b: Vec<u8>,
    ) -> Result<(), WriteError> {
        self.write(level, inst, option, dim, Layer::Value, Value::Bytes(b))
    }

    /// Set a **weight** for `dimension`, shared across all options (the 99% case).
    pub fn set_weight(
        &mut self,
        level: &str,
        inst: &str,
        dim: &str,
        w: f64,
    ) -> Result<(), WriteError> {
        self.write(level, inst, SHARED, dim, Layer::Weight, Value::F64(w))
    }

    /// Set a **per-option weight** — the orchestrator's gas pedal (and the home of an
    /// exploration bonus: a lower weight floats an under-measured option up the queue).
    pub fn set_option_weight(
        &mut self,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        w: f64,
    ) -> Result<(), WriteError> {
        self.write(level, inst, option, dim, Layer::Weight, Value::F64(w))
    }

    /// A stricter writer at `level` or finer (POLA / macaroon attenuation).
    pub fn attenuate(&mut self, level: &str) -> Option<Writer<'_>> {
        let floor = self.tensor.level_index(level)?.max(self.floor);
        Some(Writer {
            tensor: self.tensor,
            floor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cursor(pairs: &[(&str, &str)]) -> Cursor {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn fold_resolve_and_write_down() {
        let mut t = Tensor::new(["global", "workspace", "device"]);
        let mut o = t.writer("global").unwrap();
        o.set_value("global", "", "groq", "latency", 0.4).unwrap();
        o.set_value("global", "", "cerebras", "latency", 0.3)
            .unwrap();
        o.set_weight("global", "", "latency", 1.0).unwrap();
        assert_eq!(t.resolve(&Cursor::new()), vec!["cerebras", "groq"]);

        t.writer("device")
            .unwrap()
            .set_value("device", "A", "groq", "latency", 0.05)
            .unwrap();
        assert_eq!(
            t.resolve(&cursor(&[("device", "A")])),
            vec!["groq", "cerebras"]
        );
        assert_eq!(
            t.writer("device")
                .unwrap()
                .set_value("global", "", "groq", "latency", 0.0),
            Err(WriteError::WriteUp)
        );
    }

    #[test]
    fn per_option_weight_is_the_gas_pedal() {
        let mut t = Tensor::new(["global"]);
        let mut o = t.writer("global").unwrap();
        o.set_value("global", "", "groq", "latency", 0.2).unwrap();
        o.set_value("global", "", "cerebras", "latency", 0.3)
            .unwrap();
        o.set_weight("global", "", "latency", 1.0).unwrap();
        assert_eq!(t.resolve(&Cursor::new()), vec!["groq", "cerebras"]);
        t.writer("global")
            .unwrap()
            .set_option_weight("global", "", "groq", "latency", 10.0)
            .unwrap();
        assert_eq!(t.resolve(&Cursor::new()), vec!["cerebras", "groq"]);
    }

    #[test]
    fn corroborate_needs_independent_reporters() {
        // Three devices each attest that subject "D" can scrape; "D" also vouches for
        // itself. With quorum 2, the self-vote is excluded but A,B,C still corroborate.
        let mut t = Tensor::new(["global", "device"]);
        for r in ["A", "B", "C"] {
            t.writer("device")
                .unwrap()
                .set_value("device", r, "D", "can_scrape", 1.0)
                .unwrap();
        }
        // a faker "E" claims a capability only it vouches for.
        t.writer("device")
            .unwrap()
            .set_value("device", "E", "E", "can_magic", 1.0)
            .unwrap();
        // D's self-vote (should be excluded).
        t.writer("device")
            .unwrap()
            .set_value("device", "D", "D", "can_scrape", 1.0)
            .unwrap();

        t.corroborate("device", "global", "", 2);
        // D's capability is corroborated by 3 third parties.
        assert_eq!(t.value(&Cursor::new(), "D", "can_scrape"), Some(1.0));
        // E's self-claim has 0 independent reporters → not materialized.
        assert_eq!(t.value(&Cursor::new(), "E", "can_magic"), None);
    }

    #[test]
    fn reduce_median_rolls_value_up() {
        let mut t = Tensor::new(["global", "device"]);
        t.writer("device")
            .unwrap()
            .set_value("device", "A", "groq", "latency", 0.2)
            .unwrap();
        t.writer("device")
            .unwrap()
            .set_value("device", "B", "groq", "latency", 0.4)
            .unwrap();
        t.roll_up("device", "global", "");
        assert_eq!(
            t.value(&Cursor::new(), "groq", "latency"),
            Some(0.30000000000000004) // even-median average; identical bits in TS
        );
    }

    #[test]
    fn reduce_sum_adds_value_up() {
        let mut t = Tensor::new(["global", "device"]);
        t.writer("device")
            .unwrap()
            .set_value("device", "A", "groq", "tokens", 200.0)
            .unwrap();
        t.writer("device")
            .unwrap()
            .set_value("device", "B", "groq", "tokens", 350.0)
            .unwrap();
        t.reduce_sum("device", "global", "", "groq", "tokens");
        // Total draw on a shared pool is the SUM (the budget feeder), not the median.
        assert_eq!(t.value(&Cursor::new(), "groq", "tokens"), Some(550.0));
    }

    #[test]
    fn non_finite_value_gates_the_option() {
        let mut t = Tensor::new(["global", "device"]);
        let mut w = t.writer("global").unwrap();
        w.set_weight("global", "", "latency", 1.0).unwrap();
        w.set_value("global", "", "keep", "latency", 0.2).unwrap();
        w.set_value("global", "", "pos", "latency", 0.1).unwrap();
        w.set_value("global", "", "neg", "latency", 0.05).unwrap();
        w.set_value("global", "", "nanopt", "latency", 0.04)
            .unwrap();
        // A non-finite value on ANY dim gates the option — +inf, -inf, and NaN alike.
        w.set_value("global", "", "pos", "reach", f64::INFINITY)
            .unwrap();
        w.set_value("global", "", "neg", "reach", f64::NEG_INFINITY)
            .unwrap();
        w.set_value("global", "", "nanopt", "reach", f64::NAN)
            .unwrap();
        // Only the finite option survives: -inf does not anti-gate, NaN does not linger.
        assert_eq!(t.resolve(&Cursor::new()), vec!["keep"]);
    }

    #[test]
    fn a_document_rides_the_tensor_and_is_inert() {
        let mut t = Tensor::new(["global", "workspace"]);
        let mut w = t.writer("workspace").unwrap();
        let doc = vec![0u8, 1, 2, 255, 60, 62];
        w.set_bytes("workspace", "acme", "doc:cv", "state", doc.clone())
            .unwrap();
        w.set_value("workspace", "acme", "groq", "latency", 0.2)
            .unwrap();
        w.set_weight("workspace", "acme", "latency", 1.0).unwrap();
        let at = cursor(&[("workspace", "acme")]);
        assert_eq!(t.bytes(&at, "doc:cv", "state"), Some(doc.as_slice()));
        assert_eq!(t.value(&at, "doc:cv", "state"), None);
        assert_eq!(t.resolve(&at), vec!["groq"]);
    }

    #[test]
    fn json_round_trips_a_snapshot() {
        let mut a = Tensor::new(["global", "device"]);
        let mut o = a.writer("global").unwrap();
        o.set_value("global", "", "groq", "latency", 0.4).unwrap();
        o.set_weight("global", "", "latency", 1.0).unwrap();
        o.set_option_weight("global", "", "groq", "latency", 2.0)
            .unwrap();
        a.writer("device")
            .unwrap()
            .set_value("device", "A", "peer:X", "reach", f64::INFINITY)
            .unwrap();
        assert!(a.writer("workspace").is_none()); // unknown level

        let json = a.to_json();
        let b = Tensor::from_json(&json).unwrap();
        assert_eq!(b.value(&Cursor::new(), "groq", "latency"), Some(0.4));
        assert_eq!(b.weight(&Cursor::new(), "cerebras", "latency"), Some(1.0)); // shared
        assert_eq!(b.weight(&Cursor::new(), "groq", "latency"), Some(2.0)); // per-option
        assert_eq!(
            b.value(&cursor(&[("device", "A")]), "peer:X", "reach"),
            Some(f64::INFINITY) // gate survived
        );
        assert_eq!(b.to_json(), json); // canonical
    }

    #[test]
    fn apply_json_drops_a_cell_with_a_malformed_weight_layer_whole() {
        // PARITY: a cell with a valid `v` but a malformed `w` must be dropped WHOLE —
        // the value layer must NOT survive. Rust's serde decodes the JsonCell atomically;
        // the TS core was patched to match (decode both layers, then put). Both cores
        // must yield value(x, lat) == None for this input.
        let cells = r#"[
            {"level":"global","inst":"","option":"x","dim":"lat","v":0.5,"w":true}
        ]"#;
        let mut t = Tensor::new(["global"]);
        t.apply_json(cells);
        assert_eq!(t.value(&Cursor::new(), "x", "lat"), None);
        assert_eq!(t.weight(&Cursor::new(), "x", "lat"), None);
    }

    #[test]
    fn clear_level_inst_removes_one_slice_and_spares_the_others() {
        let mut t = Tensor::new(["global", "job"]);
        {
            let mut w = t.writer("job").unwrap();
            w.set_value("job", "i", "opt", "priv:cap", f64::INFINITY)
                .unwrap();
            w.set_value("job", "i", "opt", "lat", 0.2).unwrap();
            w.set_value("job", "j", "opt", "lat", 0.3).unwrap();
        }
        // The (job, i) slice exists.
        let mut at_i = Cursor::new();
        at_i.insert("job".into(), "i".into());
        assert_eq!(t.value(&at_i, "opt", "lat"), Some(0.2));

        t.clear_level_inst("job", "i");
        // (job, i) is gone …
        assert_eq!(t.value(&at_i, "opt", "lat"), None);
        assert_eq!(t.value(&at_i, "opt", "priv:cap"), None);
        // … but (job, j) survives.
        let mut at_j = Cursor::new();
        at_j.insert("job".into(), "j".into());
        assert_eq!(t.value(&at_j, "opt", "lat"), Some(0.3));
    }

    #[test]
    fn apply_json_tolerates_a_malformed_cell() {
        let cells = r#"[
            {"level":"global","inst":"","option":"groq","dim":"latency","v":0.1},
            {"level":"global","inst":"","option":"bad","dim":"latency","v":true},
            {"level":"global","inst":"","option":"cerebras","dim":"latency","v":0.2}
        ]"#;
        let mut t = Tensor::new(["global"]);
        t.apply_json(cells);
        assert_eq!(t.value(&Cursor::new(), "groq", "latency"), Some(0.1));
        assert_eq!(t.value(&Cursor::new(), "cerebras", "latency"), Some(0.2));
        assert_eq!(t.value(&Cursor::new(), "bad", "latency"), None);
    }
}
