// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Declarative construction of a stacked tensor.

use crate::tensor::{Scope, Tensor, Value, WriteError, Writer, SHARED};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeedKind {
    Value,
    Bytes,
    Weight,
    OptionWeight,
}

/// One declarative write into one scoped preference layer.
#[derive(Clone, Debug, PartialEq)]
pub struct SeedCell {
    pub floor: String,
    pub layer: String,
    pub scope: Scope,
    pub option: String,
    pub dim: String,
    pub kind: SeedKind,
    pub value: Value,
}

impl SeedCell {
    pub fn value(
        floor: &str,
        layer: &str,
        scope: Scope,
        option: &str,
        dim: &str,
        value: f64,
    ) -> Self {
        SeedCell {
            floor: floor.into(),
            layer: layer.into(),
            scope,
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::Value,
            value: Value::F64(value),
        }
    }

    pub fn bytes(
        floor: &str,
        layer: &str,
        scope: Scope,
        option: &str,
        dim: &str,
        value: Vec<u8>,
    ) -> Self {
        SeedCell {
            floor: floor.into(),
            layer: layer.into(),
            scope,
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::Bytes,
            value: Value::Bytes(value),
        }
    }

    pub fn weight(floor: &str, layer: &str, scope: Scope, dim: &str, value: f64) -> Self {
        SeedCell {
            floor: floor.into(),
            layer: layer.into(),
            scope,
            option: SHARED.into(),
            dim: dim.into(),
            kind: SeedKind::Weight,
            value: Value::F64(value),
        }
    }

    pub fn option_weight(
        floor: &str,
        layer: &str,
        scope: Scope,
        option: &str,
        dim: &str,
        value: f64,
    ) -> Self {
        SeedCell {
            floor: floor.into(),
            layer: layer.into(),
            scope,
            option: option.into(),
            dim: dim.into(),
            kind: SeedKind::OptionWeight,
            value: Value::F64(value),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorConfig {
    pub layers: Vec<String>,
    pub seeds: Vec<SeedCell>,
}

impl TensorConfig {
    pub fn new(layers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        TensorConfig {
            layers: layers.into_iter().map(Into::into).collect(),
            seeds: Vec::new(),
        }
    }

    pub fn with_cell(mut self, cell: SeedCell) -> Self {
        self.seeds.push(cell);
        self
    }
}

pub fn instantiate(config: &TensorConfig) -> Result<Tensor, WriteError> {
    let mut tensor = Tensor::new(config.layers.clone());
    for cell in &config.seeds {
        let mut writer = tensor.writer(&cell.floor).ok_or(WriteError::UnknownLayer)?;
        apply_cell(&mut writer, cell)?;
    }
    Ok(tensor)
}

fn apply_cell(writer: &mut Writer<'_>, cell: &SeedCell) -> Result<(), WriteError> {
    match &cell.kind {
        SeedKind::Value => writer.set_value(
            &cell.layer,
            &cell.scope,
            &cell.option,
            &cell.dim,
            cell.value.as_f64().ok_or(WriteError::BadValueKind)?,
        ),
        SeedKind::Bytes => writer.set_bytes(
            &cell.layer,
            &cell.scope,
            &cell.option,
            &cell.dim,
            cell.value
                .as_bytes()
                .ok_or(WriteError::BadValueKind)?
                .to_vec(),
        ),
        SeedKind::Weight => writer.set_weight(
            &cell.layer,
            &cell.scope,
            &cell.dim,
            cell.value.as_f64().ok_or(WriteError::BadValueKind)?,
        ),
        SeedKind::OptionWeight => writer.set_option_weight(
            &cell.layer,
            &cell.scope,
            &cell.option,
            &cell.dim,
            cell.value.as_f64().ok_or(WriteError::BadValueKind)?,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{scope, Cursor};

    #[test]
    fn declarative_config_preserves_scoped_layer_preferences() {
        let config = TensorConfig::new(["workspace", "stage", "consumer"])
            .with_cell(SeedCell::weight(
                "workspace",
                "workspace",
                Scope::new(),
                "quality",
                1.0,
            ))
            .with_cell(SeedCell::value(
                "workspace",
                "workspace",
                Scope::new(),
                "groq",
                "quality",
                0.2,
            ))
            .with_cell(SeedCell::value(
                "consumer",
                "consumer",
                scope([("stage", "evaluate"), ("consumer", "fit")]),
                "groq",
                "quality",
                0.8,
            ));
        let tensor = instantiate(&config).unwrap();
        assert_eq!(
            tensor.value(
                &scope([("stage", "evaluate"), ("consumer", "fit")]),
                "groq",
                "quality"
            ),
            Some(0.8)
        );
        assert_eq!(tensor.value(&Cursor::new(), "groq", "quality"), Some(0.2));
    }

    #[test]
    fn config_enforces_write_down_and_scope_depth() {
        let write_up = TensorConfig::new(["session", "source"]).with_cell(SeedCell::value(
            "source",
            "session",
            Scope::new(),
            "browser",
            "cost",
            0.0,
        ));
        assert_eq!(instantiate(&write_up), Err(WriteError::WriteUp));

        let bad_scope =
            TensorConfig::new(["session", "source", "device"]).with_cell(SeedCell::value(
                "source",
                "source",
                scope([("device", "d1")]),
                "browser",
                "cost",
                0.0,
            ));
        assert_eq!(instantiate(&bad_scope), Err(WriteError::ScopeOutsideLayer));
    }

    #[test]
    fn bad_payload_kind_is_rejected() {
        let config = TensorConfig::new(["workspace"]).with_cell(SeedCell {
            floor: "workspace".into(),
            layer: "workspace".into(),
            scope: Scope::new(),
            option: "groq".into(),
            dim: "quality".into(),
            kind: SeedKind::Value,
            value: Value::Bytes(vec![1]),
        });
        assert_eq!(instantiate(&config), Err(WriteError::BadValueKind));
    }
}
