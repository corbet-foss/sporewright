// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Mesh roles — the thin device side of the tensor.
//!
//! Sync itself is **not here**: state moves `device → API → DB` and back as cells
//! (`Tensor::cells`/`apply_json`), versioned by the DB, carried by a lease/poll
//! transport port. This module is just the two generic device operations on top of
//! a [`Writer`]: **observe** a measured value, **gate** an option's reachability,
//! and **attest** a subject's capability (the input `corroborate` rolls up).
//!
//! **Peer-as-option.** A reachable peer is an *option* (`"peer:A"`); its
//! reachability is the [`REACH`] gate on the routing device's own slice — `+∞` when
//! unreachable (dropped by `resolve`), `0`/absent when reachable. The links that
//! route *work* are P2P; the links that sync *state* are the star to the orchestrator.

use crate::tensor::{Tensor, WriteError, Writer};

/// The reachability gate dimension for a peer option: `+∞` ⇒ unreachable now
/// (dropped by `resolve`); absent or `0` ⇒ reachable.
pub const REACH: &str = "reach";

/// The capability-gate dimension prefix: a gate on `priv:<cap>` carries `+∞` when an
/// option **lacks** a required capability `<cap>` (dropped by `resolve`); absent or
/// `0` ⇒ the option satisfies it. The vocabulary of `<cap>` is the product's
/// (`ip:residential`, `gpu:burst`, `browser:full`, `geo:CH`, …) — the engine never
/// learns a single capability name.
pub const GATE_PREFIX: &str = "priv:";

/// Compose the gate dimension `priv:<cap>` for a capability `cap`.
pub fn gate_dim(cap: &str) -> String {
    format!("{GATE_PREFIX}{cap}")
}

/// **gate_capabilities** — the vocabulary-agnostic capability gate. For each
/// `option` that the `offers` predicate says does **not** provide a `required`
/// capability, write `+∞` on `priv:<cap>` at `(level, inst)`, so `resolve` drops
/// that option for this slice. Options that offer every required capability are left
/// untouched (no gate ⇒ they remain candidates, ordered by their judgement dims).
///
/// This is the option-axis feasibility primitive of `MODEL.md` §3(C): the indicator
/// of the feasible set, `0`/absent when allowed, `+∞` when not. It is **purely
/// additive** — it only ever *adds* gates, never clears them — and it knows nothing
/// about any specific capability; the `<cap>` strings are the product's vocabulary.
///
/// `writer` must be able to write at `level` (write-down); a finer-than-floor `level`
/// is fine. Returns the first [`WriteError`] (a write-up or unknown level) it hits, or
/// `Ok(())` when every gate landed.
pub fn gate_capabilities<'a>(
    writer: &mut Writer<'a>,
    level: &str,
    inst: &str,
    options: &[&str],
    required: &[&str],
    offers: impl Fn(&str, &str) -> bool,
) -> Result<(), WriteError> {
    for cap in required {
        let dim = gate_dim(cap);
        for option in options {
            if !offers(option, cap) {
                writer.set_value(level, inst, option, &dim, f64::INFINITY)?;
            }
        }
    }
    Ok(())
}

/// A **device** in the mesh: a [`Writer`] pinned to its own slice, with the device
/// operations on top. It owns exactly its `(level, inst)` slice (write-down keeps it
/// there), so devices never conflict. The orchestrator then `reduce`s the
/// observations and `corroborate`s the attestations.
pub struct Device<'a> {
    writer: Writer<'a>,
    level: String,
    inst: String,
}

impl<'a> Device<'a> {
    /// A device handle at `level` (e.g. `"device"`) for instance `inst`. `None` if
    /// the level is unknown.
    pub fn new(t: &'a mut Tensor, level: &str, inst: &str) -> Option<Device<'a>> {
        let writer = t.writer(level)?;
        Some(Device {
            writer,
            level: level.to_string(),
            inst: inst.to_string(),
        })
    }

    /// Record a measured value for an `option` on this device's slice — the
    /// device-level "value flows up via `reduce`".
    pub fn observe(&mut self, option: &str, dim: &str, value: f64) -> Result<(), WriteError> {
        self.writer
            .set_value(&self.level, &self.inst, option, dim, value)
    }

    /// Set whether an `option` is reachable from this device now: `false` gates it
    /// (`REACH = +∞`, dropped by `resolve`), `true` opens it (`0`).
    pub fn gate(&mut self, option: &str, reachable: bool) -> Result<(), WriteError> {
        let v = if reachable { 0.0 } else { f64::INFINITY };
        self.writer
            .set_value(&self.level, &self.inst, option, REACH, v)
    }

    /// Attest whether a subject `option` has capability `dim` — the boolean input
    /// the orchestrator's `corroborate` counts (independent reporters, self-excluded).
    pub fn attest(&mut self, option: &str, dim: &str, can: bool) -> Result<(), WriteError> {
        let v = if can { 1.0 } else { 0.0 };
        self.writer
            .set_value(&self.level, &self.inst, option, dim, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Cursor, Tensor};

    fn at(device: &str) -> Cursor {
        [("device".to_string(), device.to_string())]
            .into_iter()
            .collect()
    }

    #[test]
    fn the_device_mesh_loop_over_primitives() {
        // Two devices observe a shared option; the orchestrator rolls it up and
        // resolves. A third device gates an unreachable peer on its own slice.
        let mut t = Tensor::new(["global", "device"]);
        t.writer("global")
            .unwrap()
            .set_weight("global", "", "latency", 1.0)
            .unwrap();
        Device::new(&mut t, "device", "A")
            .unwrap()
            .observe("groq", "latency", 0.2)
            .unwrap();
        {
            let mut d = Device::new(&mut t, "device", "B").unwrap();
            d.observe("groq", "latency", 0.4).unwrap();
            d.gate("peer:C", false).unwrap();
        }

        t.roll_up("device", "global", "");
        assert_eq!(
            t.value(&Cursor::new(), "groq", "latency"),
            Some(0.30000000000000004)
        );
        assert_eq!(t.resolve(&at("B")), vec!["groq"]); // unreachable peer dropped
    }

    #[test]
    fn devices_attest_and_the_orchestrator_corroborates() {
        let mut t = Tensor::new(["global", "device"]);
        for r in ["A", "B"] {
            Device::new(&mut t, "device", r)
                .unwrap()
                .attest("peer:D", "can_scrape", true)
                .unwrap();
        }
        t.corroborate("device", "global", "", 2);
        assert_eq!(t.value(&Cursor::new(), "peer:D", "can_scrape"), Some(1.0));
    }

    #[test]
    fn gate_capabilities_drops_options_missing_a_required_cap() {
        // Three options with different capability sets; a job requires two caps. Only
        // the option that offers BOTH survives resolve at the job slice.
        let mut t = Tensor::new(["global", "source", "job"]);
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", "", "financial", 1.0).unwrap();
            for o in ["full", "light", "curl"] {
                w.set_value("global", "", o, "financial", 0.0).unwrap();
            }
        }
        // offers: only "full" has both browser:full and ip:residential.
        let caps = |o: &str, c: &str| -> bool {
            matches!(
                (o, c),
                ("full", "browser:full")
                    | ("full", "ip:residential")
                    | ("light", "browser:full")
                    | ("curl", "ip:residential")
            )
        };
        {
            let mut w = t.writer("global").unwrap();
            gate_capabilities(
                &mut w,
                "job",
                "j1",
                &["full", "light", "curl"],
                &["browser:full", "ip:residential"],
                caps,
            )
            .unwrap();
        }
        let mut cur = Cursor::new();
        cur.insert("job".into(), "j1".into());
        assert_eq!(t.resolve(&cur), vec!["full"]);
        // The gate is per-slice: a different job with no gates keeps the whole fleet.
        assert_eq!(
            t.resolve(
                &[("job".to_string(), "j2".to_string())]
                    .into_iter()
                    .collect()
            ),
            vec!["curl", "full", "light"] // by name, all financial 0.0
        );
    }

    #[test]
    fn gate_capabilities_no_required_caps_gates_nothing() {
        let mut t = Tensor::new(["global", "job"]);
        {
            let mut w = t.writer("global").unwrap();
            w.set_weight("global", "", "financial", 1.0).unwrap();
            w.set_value("global", "", "a", "financial", 0.0).unwrap();
            w.set_value("global", "", "b", "financial", 0.0).unwrap();
        }
        {
            let mut w = t.writer("global").unwrap();
            // offers() is never consulted when there are no required caps.
            gate_capabilities(&mut w, "job", "j", &["a", "b"], &[], |_, _| false).unwrap();
        }
        assert_eq!(
            t.resolve(&[("job".to_string(), "j".to_string())].into_iter().collect()),
            vec!["a", "b"]
        );
    }

    #[test]
    fn gate_capabilities_refuses_to_write_up() {
        // A writer floored at "job" cannot gate at the coarser "global" level.
        let mut t = Tensor::new(["global", "job"]);
        let mut w = t.writer("job").unwrap();
        let e = gate_capabilities(&mut w, "global", "", &["a"], &["x"], |_, _| false);
        assert_eq!(e, Err(WriteError::WriteUp));
    }

    #[test]
    fn peer_is_an_option_gated_by_reachability() {
        let mut t = Tensor::new(["global", "device"]);
        let mut o = t.writer("global").unwrap();
        o.set_value("global", "", "local", "latency", 0.5).unwrap();
        o.set_value("global", "", "peer:A", "latency", 0.2).unwrap();
        o.set_weight("global", "", "latency", 1.0).unwrap();

        Device::new(&mut t, "device", "D")
            .unwrap()
            .gate("peer:A", false)
            .unwrap();
        assert_eq!(t.resolve(&at("D")), vec!["local"]);

        Device::new(&mut t, "device", "D")
            .unwrap()
            .gate("peer:A", true)
            .unwrap();
        assert_eq!(t.resolve(&at("D")), vec!["peer:A", "local"]);
    }
}
