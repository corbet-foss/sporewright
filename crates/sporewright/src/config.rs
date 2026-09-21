// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! **Declarative tensor config** — a product declares its tensor as *data*.
//!
//! `MODEL.md` §7 / `ROUTING-MODEL.md` §7 draw the library boundary as **facts in →
//! decision out**: a product supplies its *levels*, *options*, *dims*, *weights*, and
//! *gates* (all data the engine routes over) and writes no routing logic. Today a
//! product expresses that seeding **imperatively** — a sequence of [`Writer`] calls
//! (e.g. a fleet seed, a cascade seed). This module lets the same seeding be
//! a plain [`TensorConfig`] value (levels + a list of seed cells) that
//! [`instantiate`] turns into a [`Tensor`]. The product declares its tensor as DATA,
//! not code — the SCOPE split made literal.
//!
//! **Faithful to the imperative path, not a parallel one.** A [`SeedCell`] is exactly
//! one [`Writer`] call: it carries a **`floor`** (the tier the minting writer is
//! granted) plus the write coordinate and the [`Layer`] kind. [`instantiate`] mints a
//! `writer(floor)` per cell and applies it, so **write-down is enforced identically**
//! — a config whose cell writes *coarser* than its `floor` is rejected with the same
//! [`WriteError::WriteUp`] the imperative call would raise, and an unknown level with
//! [`WriteError::UnknownLevel`]. There is no second write path: a config IS a list of
//! `Writer` calls, just reified as data. Expressing an existing imperative seeding as
//! a `TensorConfig` and instantiating it yields a tensor **byte-identical** to the
//! imperatively-built one (the round-trip test below, mirrored in TS).
//!
//! **No engine change, decision-equivalent.** `instantiate` only *writes* cells the
//! `Writer` already writes; `resolve`/`reduce`/`fold` are untouched. The Rust and TS
//! builders apply cells in the **same declared order**, so the resulting tensors are
//! identical and any later `resolve` is decision-equivalent. Gates need no special
//! kind: a gate is a non-finite **value** ([`SeedCell::value`] with `+∞`/`NaN`), so it
//! seeds through the ordinary value path and `resolve` drops it exactly as before.

use crate::tensor::{Tensor, Value, WriteError, Writer};

/// Which layer a [`SeedCell`] writes — the data twin of the (internal) write kind.
/// Mirrors the four [`Writer`] setters: a scalar value, a document blob, an
/// option-shared weight, or a per-option (gas-pedal) weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeedKind {
    /// A scalar **value** (`Writer::set_value`). A non-finite value (`+∞`/`-∞`/`NaN`)
    /// is a **gate** — no special kind, it rides the value path and `resolve` drops it.
    Value,
    /// A **document** blob (`Writer::set_bytes`) — inert to the routing math.
    Bytes,
    /// An **option-shared weight** (`Writer::set_weight`) — the 99% weight case, set at
    /// the shared `""` slot and inherited by every contender. The cell's `option` is
    /// ignored (the setter always writes the shared slot).
    Weight,
    /// A **per-option weight** (`Writer::set_option_weight`) — the orchestrator's
    /// gas pedal, pinned to one `(option, dim)`.
    OptionWeight,
}

/// One declarative seed — exactly one [`Writer`] call reified as data.
///
/// `floor` is the tier the minting writer is granted (the write-down floor); `level`
/// is the tier the cell lands at (must be `floor` or finer). For [`SeedKind::Weight`]
/// the `option` is ignored (the shared `""` slot is always used). The payload is a
/// [`Value`] so a gate (`Value::F64(f64::INFINITY)`) and a document
/// (`Value::Bytes`) both express naturally.
#[derive(Clone, Debug, PartialEq)]
pub struct SeedCell {
    /// The write-down floor — the tier the writer minted for this cell is granted.
    pub floor: String,
    /// The tier the cell lands at (must be `floor` or finer, else [`WriteError::WriteUp`]).
    pub level: String,
    /// The instance on `level` (the shared `""` slot, or a concrete instance).
    pub inst: String,
    /// The option this cell is about (ignored for [`SeedKind::Weight`]).
    pub option: String,
    /// The dimension this cell is about.
    pub dim: String,
    /// Which layer to write.
    pub kind: SeedKind,
    /// The payload — a scalar (incl. a non-finite gate) or document bytes.
    pub value: Value,
}

impl SeedCell {
    /// A scalar **value** seed (`Writer::set_value`). A non-finite `v` is a gate.
    pub fn value(floor: &str, level: &str, inst: &str, option: &str, dim: &str, v: f64) -> Self {
        SeedCell {
            floor: floor.into(),
            level: level.into(),
            inst: inst.into(),
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::Value,
            value: Value::F64(v),
        }
    }

    /// A **document** seed (`Writer::set_bytes`).
    pub fn bytes(
        floor: &str,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        b: Vec<u8>,
    ) -> Self {
        SeedCell {
            floor: floor.into(),
            level: level.into(),
            inst: inst.into(),
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::Bytes,
            value: Value::Bytes(b),
        }
    }

    /// An **option-shared weight** seed (`Writer::set_weight`). `option` is the shared
    /// `""` slot by construction.
    pub fn weight(floor: &str, level: &str, inst: &str, dim: &str, w: f64) -> Self {
        SeedCell {
            floor: floor.into(),
            level: level.into(),
            inst: inst.into(),
            option: crate::tensor::SHARED.into(),
            dim: dim.into(),
            kind: SeedKind::Weight,
            value: Value::F64(w),
        }
    }

    /// A **per-option weight** seed — the gas pedal (`Writer::set_option_weight`).
    pub fn option_weight(
        floor: &str,
        level: &str,
        inst: &str,
        option: &str,
        dim: &str,
        w: f64,
    ) -> Self {
        SeedCell {
            floor: floor.into(),
            level: level.into(),
            inst: inst.into(),
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::OptionWeight,
            value: Value::F64(w),
        }
    }
}

/// A product's tensor declared as **data**: the ordered context-tier skeleton plus the
/// seed cells, applied in declared order. Plain data — no `Writer`, no engine state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorConfig {
    /// The context-tier skeleton, coarsest first — exactly [`Tensor::new`]'s argument.
    pub levels: Vec<String>,
    /// The seed cells, applied **in this order** (so a later cell overwrites an earlier
    /// one at the same coordinate, exactly as repeated imperative writes would).
    pub seeds: Vec<SeedCell>,
}

impl TensorConfig {
    /// A config over a fixed, ordered context-tier skeleton (no seeds yet).
    pub fn new(levels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        TensorConfig {
            levels: levels.into_iter().map(Into::into).collect(),
            seeds: Vec::new(),
        }
    }

    /// Append a seed cell (builder style). Order is preserved — it is load-bearing for
    /// overwrite semantics and cross-core equivalence.
    pub fn with_cell(mut self, cell: SeedCell) -> Self {
        self.seeds.push(cell);
        self
    }
}

/// Build a [`Tensor`] from a [`TensorConfig`]: create the tier skeleton, then for each
/// seed mint a `writer(floor)` and apply the cell — the **same write-down-enforced
/// path** the imperative seeding takes. Cells are applied in declared order.
///
/// Returns the first [`WriteError`] verbatim — one of three rejection causes: an unknown
/// `floor` or `level` ([`WriteError::UnknownLevel`]), a cell that would write coarser than
/// its `floor` ([`WriteError::WriteUp`]), or a payload whose declared kind disagrees with
/// its value type ([`WriteError::BadValueKind`] — e.g. a `Bytes` cell carrying an `F64`).
/// On success the tensor is byte-identical to the same sequence of imperative `Writer`
/// calls.
pub fn instantiate(config: &TensorConfig) -> Result<Tensor, WriteError> {
    let mut t = Tensor::new(config.levels.clone());
    for cell in &config.seeds {
        // Mint a writer granted at the cell's floor — an unknown floor is the same
        // error the imperative `t.writer(floor)` would surface (None → UnknownLevel).
        let mut w: Writer<'_> = t.writer(&cell.floor).ok_or(WriteError::UnknownLevel)?;
        apply_cell(&mut w, cell)?;
    }
    Ok(t)
}

/// Apply one seed through a [`Writer`] — the single dispatch from a [`SeedKind`] to its
/// [`Writer`] setter. Write-down (and unknown-level) is enforced by the setter itself.
fn apply_cell(w: &mut Writer<'_>, cell: &SeedCell) -> Result<(), WriteError> {
    // A kind/value-payload mismatch (e.g. `SeedKind::Bytes` carrying an `F64`, or a
    // scalar kind carrying `Bytes`) is a malformed CELL, not an unknown TIER — surface
    // it as `BadValueKind` so the author is pointed at the cell, not the levels list.
    match &cell.kind {
        SeedKind::Value => {
            let v = cell.value.as_f64().ok_or(WriteError::BadValueKind)?;
            w.set_value(&cell.level, &cell.inst, &cell.option, &cell.dim, v)
        }
        SeedKind::Bytes => {
            let b = cell
                .value
                .as_bytes()
                .ok_or(WriteError::BadValueKind)?
                .to_vec();
            w.set_bytes(&cell.level, &cell.inst, &cell.option, &cell.dim, b)
        }
        SeedKind::Weight => {
            let v = cell.value.as_f64().ok_or(WriteError::BadValueKind)?;
            w.set_weight(&cell.level, &cell.inst, &cell.dim, v)
        }
        SeedKind::OptionWeight => {
            let v = cell.value.as_f64().ok_or(WriteError::BadValueKind)?;
            w.set_option_weight(&cell.level, &cell.inst, &cell.option, &cell.dim, v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Cursor, Tensor};

    /// The shape of a product fleet seed, expressed as a
    /// `TensorConfig`. A product seeds judgement weights at the GLOBAL shared
    /// slot, then one `financial` value per device. We reproduce that *shape* — the
    /// faithfulness claim is "a config of these cells == those imperative `Writer` calls".
    const GLOBAL: &str = "global";
    const SOURCE: &str = "source";
    const JOB: &str = "job";
    const FINANCIAL: &str = "financial";
    const LATENCY: &str = "latency";
    const RELIABILITY: &str = "reliability";

    /// Imperatively seed a fleet exactly as `seed_fleet` does.
    fn seed_fleet_imperative(t: &mut Tensor, fleet: &[(&str, f64)]) {
        let mut w = t.writer(GLOBAL).expect("global writer");
        w.set_weight(GLOBAL, "", FINANCIAL, 1.0).unwrap();
        w.set_weight(GLOBAL, "", LATENCY, 0.001).unwrap();
        w.set_weight(GLOBAL, "", RELIABILITY, 4.0).unwrap();
        for (id, financial) in fleet {
            w.set_value(GLOBAL, "", id, FINANCIAL, *financial).unwrap();
        }
    }

    /// The SAME seeding as a `TensorConfig`, cell for cell, in the SAME order.
    fn seed_fleet_config(fleet: &[(&str, f64)]) -> TensorConfig {
        let mut cfg = TensorConfig::new([GLOBAL, SOURCE, JOB])
            .with_cell(SeedCell::weight(GLOBAL, GLOBAL, "", FINANCIAL, 1.0))
            .with_cell(SeedCell::weight(GLOBAL, GLOBAL, "", LATENCY, 0.001))
            .with_cell(SeedCell::weight(GLOBAL, GLOBAL, "", RELIABILITY, 4.0));
        for (id, financial) in fleet {
            cfg = cfg.with_cell(SeedCell::value(
                GLOBAL, GLOBAL, "", id, FINANCIAL, *financial,
            ));
        }
        cfg
    }

    fn fleet() -> Vec<(&'static str, f64)> {
        vec![
            ("dev:browser:b1", 0.0),
            ("dev:container:c1", 0.1),
            ("dev:render:r1", 1.0),
        ]
    }

    #[test]
    fn config_instantiation_is_byte_identical_to_imperative_seeding() {
        // FAITHFULNESS: the declarative builder produces a tensor byte-identical to the
        // imperative `seed_fleet`. The wire form (`to_json`) is the byte witness.
        let f = fleet();
        let mut imperative = Tensor::new([GLOBAL, SOURCE, JOB]);
        seed_fleet_imperative(&mut imperative, &f);

        let declared = instantiate(&seed_fleet_config(&f)).expect("config instantiates");

        assert_eq!(
            declared.to_json(),
            imperative.to_json(),
            "declared tensor must be byte-identical to the imperative seeding"
        );
    }

    #[test]
    fn declared_and_imperative_resolve_identically() {
        // DECISION-EQUIVALENCE within the core: the two tensors route the same.
        let f = fleet();
        let mut imperative = Tensor::new([GLOBAL, SOURCE, JOB]);
        seed_fleet_imperative(&mut imperative, &f);
        let declared = instantiate(&seed_fleet_config(&f)).unwrap();

        // financial weight 1.0 dominates; free browser first, paid render last.
        assert_eq!(
            imperative.resolve(&Cursor::new()),
            declared.resolve(&Cursor::new())
        );
        assert_eq!(
            declared.resolve(&Cursor::new()),
            vec!["dev:browser:b1", "dev:container:c1", "dev:render:r1"]
        );
    }

    #[test]
    fn all_four_seed_kinds_round_trip_through_a_config() {
        // value, bytes, shared weight, and per-option weight — each kind must land
        // exactly where the matching imperative setter would put it.
        let doc = vec![0u8, 1, 2, 255, 60, 62];
        let cfg = TensorConfig::new([GLOBAL, "workspace"])
            .with_cell(SeedCell::value(GLOBAL, GLOBAL, "", "groq", "latency", 0.4))
            .with_cell(SeedCell::weight(GLOBAL, GLOBAL, "", "latency", 1.0))
            .with_cell(SeedCell::option_weight(
                GLOBAL, GLOBAL, "", "groq", "latency", 2.0,
            ))
            .with_cell(SeedCell::bytes(
                "workspace",
                "workspace",
                "acme",
                "doc:cv",
                "state",
                doc.clone(),
            ));

        // The imperative equivalent.
        let mut imp = Tensor::new([GLOBAL, "workspace"]);
        {
            let mut w = imp.writer(GLOBAL).unwrap();
            w.set_value(GLOBAL, "", "groq", "latency", 0.4).unwrap();
            w.set_weight(GLOBAL, "", "latency", 1.0).unwrap();
            w.set_option_weight(GLOBAL, "", "groq", "latency", 2.0)
                .unwrap();
        }
        imp.writer("workspace")
            .unwrap()
            .set_bytes("workspace", "acme", "doc:cv", "state", doc.clone())
            .unwrap();

        let built = instantiate(&cfg).unwrap();
        assert_eq!(built.to_json(), imp.to_json());

        // and the document is inert / readable as the imperative path leaves it.
        let mut at = Cursor::new();
        at.insert("workspace".into(), "acme".into());
        assert_eq!(built.bytes(&at, "doc:cv", "state"), Some(doc.as_slice()));
        assert_eq!(built.value(&Cursor::new(), "groq", "latency"), Some(0.4));
        assert_eq!(built.weight(&Cursor::new(), "groq", "latency"), Some(2.0)); // per-option
    }

    #[test]
    fn a_gate_seeds_through_the_value_path() {
        // A gate is just a non-finite value — no special kind.
        let cfg = TensorConfig::new([GLOBAL, JOB])
            .with_cell(SeedCell::weight(GLOBAL, GLOBAL, "", "latency", 1.0))
            .with_cell(SeedCell::value(GLOBAL, GLOBAL, "", "keep", "latency", 0.2))
            .with_cell(SeedCell::value(GLOBAL, GLOBAL, "", "drop", "latency", 0.1))
            .with_cell(SeedCell::value(
                JOB,
                JOB,
                "j1",
                "drop",
                "priv:residential",
                f64::INFINITY,
            ));
        let t = instantiate(&cfg).unwrap();
        let mut at = Cursor::new();
        at.insert(JOB.into(), "j1".into());
        // the +∞ gate drops "drop" for this job; "keep" survives.
        assert_eq!(t.resolve(&at), vec!["keep"]);
    }

    #[test]
    fn a_write_up_cell_is_rejected_like_the_imperative_call() {
        // floor=job (fine) but the cell writes at global (coarse) → WriteUp, exactly
        // as `t.writer("job").set_value("global", ...)` would.
        let cfg = TensorConfig::new([GLOBAL, JOB])
            .with_cell(SeedCell::value(JOB, GLOBAL, "", "x", "latency", 0.1));
        assert_eq!(instantiate(&cfg), Err(WriteError::WriteUp));
    }

    #[test]
    fn a_kind_value_mismatch_is_rejected_as_bad_value_kind() {
        // A cell whose declared kind disagrees with its value payload (here `Bytes`
        // carrying an f64) is a malformed CELL, not an unknown TIER → BadValueKind, NOT
        // UnknownLevel. The tiers are all valid.
        let cfg = TensorConfig::new([GLOBAL]).with_cell(SeedCell {
            floor: GLOBAL.into(),
            level: GLOBAL.into(),
            inst: "".into(),
            option: "x".into(),
            dim: "doc".into(),
            kind: SeedKind::Bytes,
            value: Value::F64(1.0), // wrong payload type for a Bytes kind
        });
        assert_eq!(instantiate(&cfg), Err(WriteError::BadValueKind));
    }

    #[test]
    fn an_unknown_level_is_rejected() {
        // an unknown floor surfaces UnknownLevel (the `t.writer(floor)` → None case).
        let cfg = TensorConfig::new([GLOBAL])
            .with_cell(SeedCell::value("nope", "nope", "", "x", "latency", 0.1));
        assert_eq!(instantiate(&cfg), Err(WriteError::UnknownLevel));
        // an unknown target level (with a valid floor) is the setter's UnknownLevel.
        let cfg2 = TensorConfig::new([GLOBAL])
            .with_cell(SeedCell::value(GLOBAL, "missing", "", "x", "latency", 0.1));
        assert_eq!(instantiate(&cfg2), Err(WriteError::UnknownLevel));
    }
}
