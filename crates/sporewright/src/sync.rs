// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Participant-side operations over a scoped preference layer.

use crate::tensor::{Scope, Tensor, WriteError, Writer};

pub const REACH: &str = "reach";
pub const GATE_PREFIX: &str = "priv:";

pub fn gate_dim(capability: &str) -> String {
    format!("{GATE_PREFIX}{capability}")
}

pub fn gate_capabilities(
    writer: &mut Writer<'_>,
    layer: &str,
    scope: &Scope,
    options: &[&str],
    required: &[&str],
    offers: impl Fn(&str, &str) -> bool,
) -> Result<(), WriteError> {
    for capability in required {
        let dimension = gate_dim(capability);
        for option in options {
            if !offers(option, capability) {
                writer.set_value(layer, scope, option, &dimension, f64::INFINITY)?;
            }
        }
    }
    Ok(())
}

/// A participant pinned to one exact layer-owned preference scope.
pub struct Device<'a> {
    writer: Writer<'a>,
    layer: String,
    scope: Scope,
}

impl<'a> Device<'a> {
    pub fn new(tensor: &'a mut Tensor, layer: &str, scope: Scope) -> Option<Device<'a>> {
        let writer = tensor.writer(layer)?;
        Some(Device {
            writer,
            layer: layer.to_owned(),
            scope,
        })
    }

    pub fn observe(&mut self, option: &str, dimension: &str, value: f64) -> Result<(), WriteError> {
        self.writer
            .set_value(&self.layer, &self.scope, option, dimension, value)
    }

    pub fn revise(
        &mut self,
        option: &str,
        dimension: &str,
        observation: f64,
        rate: f64,
    ) -> Result<crate::tensor::PreferenceChange, WriteError> {
        self.writer.nudge_value(
            &self.layer,
            &self.scope,
            option,
            dimension,
            observation,
            rate,
        )
    }

    pub fn gate(&mut self, option: &str, reachable: bool) -> Result<(), WriteError> {
        self.writer.set_value(
            &self.layer,
            &self.scope,
            option,
            REACH,
            if reachable { 0.0 } else { f64::INFINITY },
        )
    }

    pub fn attest(
        &mut self,
        option: &str,
        dimension: &str,
        capable: bool,
    ) -> Result<(), WriteError> {
        self.writer.set_value(
            &self.layer,
            &self.scope,
            option,
            dimension,
            if capable { 1.0 } else { 0.0 },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::scope;

    #[test]
    fn local_reachability_is_conditioned_by_parent_scope() {
        let mut tensor = Tensor::new(["session", "source", "device"]);
        {
            let mut writer = tensor.writer("session").unwrap();
            writer
                .set_weight("session", &Scope::new(), "latency", 1.0)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "browser", "latency", 0.1)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "native", "latency", 0.2)
                .unwrap();
        }
        Device::new(
            &mut tensor,
            "device",
            scope([("source", "nzz"), ("device", "A")]),
        )
        .unwrap()
        .gate("browser", false)
        .unwrap();

        assert_eq!(
            tensor.resolve(&scope([("source", "nzz"), ("device", "A")])),
            vec!["native"]
        );
        assert_eq!(
            tensor.resolve(&scope([("source", "proton"), ("device", "A")])),
            vec!["browser", "native"]
        );
    }

    #[test]
    fn capability_gate_is_one_preference_in_the_target_layer() {
        let mut tensor = Tensor::new(["session", "source", "job"]);
        {
            let mut writer = tensor.writer("session").unwrap();
            writer
                .set_weight("session", &Scope::new(), "financial", 1.0)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "browser", "financial", 0.0)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "native", "financial", 0.1)
                .unwrap();
        }
        let job = scope([("source", "nzz"), ("job", "j1")]);
        gate_capabilities(
            &mut tensor.writer("job").unwrap(),
            "job",
            &job,
            &["browser", "native"],
            &["javascript"],
            |option, _| option == "browser",
        )
        .unwrap();
        assert_eq!(tensor.resolve(&job), vec!["browser"]);
    }

    #[test]
    fn corroboration_counts_distinct_scoped_reporters() {
        let mut tensor = Tensor::new(["session", "device"]);
        for reporter in ["A", "B"] {
            Device::new(&mut tensor, "device", scope([("device", reporter)]))
                .unwrap()
                .attest("peer:D", "can_scrape", true)
                .unwrap();
        }
        tensor
            .corroborate(
                "device",
                "session",
                &Scope::new(),
                ("peer:D", "can_scrape"),
                "device",
                2,
            )
            .unwrap();
        assert_eq!(
            tensor.value(&Scope::new(), "peer:D", "can_scrape"),
            Some(1.0)
        );
    }
}
