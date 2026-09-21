// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! The stacked tensor of scoped preference layers.
//!
//! A layer owns one sparse preference set. A cell is addressed by
//! `(layer, scope, option, dimension)` and holds two co-located scalars:
//! the observed/declared `value` and the governing `weight`. `scope` contains
//! owner coordinates from the layer and any of its ancestors plus free context
//! facets, so a device preference can remain conditional on the source and job
//! in which it was learned while a stage preference can vary by capability.
//!
//! A cursor selects one branch through the stack. Folding chooses the finest
//! compatible layer-local cell; resolving contracts dimensions into an ordered
//! viable option set. Feedback revises an exact layer/scope cell. Reduction is
//! the explicit, scoped projection from descendant preference sets into an
//! ancestor preference set.

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

/// The shared option slot and the curled scope value.
pub const SHARED: &str = "";

/// Conditions under which one preference applies. Keys matching layer names are
/// owner coordinates; all other keys are context facets.
pub type Scope = BTreeMap<String, String>;

/// A read cursor is the complete participant/task path currently being realized.
pub type Cursor = Scope;

/// Build a scope from `(axis, value)` pairs.
pub fn scope<const N: usize>(coordinates: [(&str, &str); N]) -> Scope {
    coordinates
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(layer, value)| (layer.to_owned(), value.to_owned()))
        .collect()
}

/// A routable scalar or an opaque document value inert to routing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    F64(f64),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(value) => Some(*value),
            Value::Bytes(_) => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(value) => Some(value),
            Value::F64(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Cell {
    value: Option<Value>,
    weight: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// A writer attempted to alter a preference layer above its authority floor.
    WriteUp,
    /// A layer name is not part of the tensor's declared stack.
    UnknownLayer,
    /// A scope pins an owner coordinate finer than the layer that owns the preference.
    ScopeOutsideLayer,
    /// A declarative seed's payload and kind disagree.
    BadValueKind,
}

mod wire_value {
    use super::Value;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Value, serializer: S) -> Result<S::Ok, S::Error> {
        match value {
            Value::F64(value) if value.is_finite() => serializer.serialize_f64(*value),
            Value::F64(value) if value.is_nan() => serializer.serialize_str("nan"),
            Value::F64(value) if *value > 0.0 => serializer.serialize_str("inf"),
            Value::F64(_) => serializer.serialize_str("-inf"),
            Value::Bytes(bytes) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("b64", &STANDARD.encode(bytes))?;
                map.end()
            }
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Number(f64),
        Tag(String),
        Bytes { b64: String },
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
        match Repr::deserialize(deserializer)? {
            Repr::Number(value) => Ok(Value::F64(value)),
            Repr::Tag(tag) => match tag.as_str() {
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

mod opt_value {
    use super::{wire_value, Value};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    struct Wrapped<'a>(&'a Value);

    impl Serialize for Wrapped<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            wire_value::serialize(self.0, serializer)
        }
    }

    pub fn serialize<S: Serializer>(
        value: &Option<Value>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&Wrapped(value)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Value>, D::Error> {
        #[derive(Deserialize)]
        struct Wrapped(#[serde(with = "wire_value")] Value);
        Ok(Option::<Wrapped>::deserialize(deserializer)?.map(|wrapped| wrapped.0))
    }
}

/// One serialized preference cell. The product's adapter owns clocks and versions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonCell {
    pub layer: String,
    #[serde(default)]
    pub scope: Scope,
    pub option: String,
    pub dim: String,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_value")]
    pub v: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_value")]
    pub w: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    layer: String,
    scope: Vec<(String, String)>,
    option: String,
    dim: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
    Value,
    Weight,
}

/// Where one effective preference came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    pub layer: String,
    pub scope: Scope,
}

/// One dimension's contribution to a resolved option.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DimensionTrace {
    pub dimension: String,
    pub value: f64,
    pub weight: Option<f64>,
    pub contribution: f64,
    pub value_origin: Origin,
    pub weight_origin: Option<Origin>,
}

/// An option considered by `resolve_explained`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedOption {
    pub option: String,
    pub viable: bool,
    pub score: f64,
    pub dimensions: Vec<DimensionTrace>,
}

/// The exact layer-local change produced by feedback or a scoped reduction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceChange {
    pub layer: String,
    pub scope: Scope,
    pub option: String,
    pub dimension: String,
    pub previous: Option<f64>,
    pub next: f64,
}

/// A sparse stack of product-declared preference layers, coarsest first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor {
    pub layers: Vec<String>,
    cells: BTreeMap<CellKey, Cell>,
}

impl Tensor {
    pub fn new(layers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Tensor {
            layers: layers.into_iter().map(Into::into).collect(),
            cells: BTreeMap::new(),
        }
    }

    fn layer_index(&self, layer: &str) -> Option<usize> {
        self.layers.iter().position(|candidate| candidate == layer)
    }

    fn normalize_scope(
        &self,
        layer: &str,
        scope: &Scope,
    ) -> Result<Vec<(String, String)>, WriteError> {
        let owner = self.layer_index(layer).ok_or(WriteError::UnknownLayer)?;
        let mut normalized = Vec::new();
        for (axis, value) in scope {
            if value.is_empty() {
                continue;
            }
            if let Some(index) = self.layer_index(axis) {
                if index > owner {
                    return Err(WriteError::ScopeOutsideLayer);
                }
            }
            normalized.push((axis.clone(), value.clone()));
        }
        normalized.sort_by(|(left, _), (right, _)| {
            self.layer_index(left)
                .unwrap_or(usize::MAX)
                .cmp(&self.layer_index(right).unwrap_or(usize::MAX))
                .then_with(|| left.cmp(right))
        });
        Ok(normalized)
    }

    fn scope_map(scope: &[(String, String)]) -> Scope {
        scope.iter().cloned().collect()
    }

    fn key(
        &self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
    ) -> Result<CellKey, WriteError> {
        Ok(CellKey {
            layer: layer.to_owned(),
            scope: self.normalize_scope(layer, scope)?,
            option: option.to_owned(),
            dim: dim.to_owned(),
        })
    }

    fn put(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        plane: Plane,
        value: Value,
    ) -> Result<(), WriteError> {
        let key = self.key(layer, scope, option, dim)?;
        let cell = self.cells.entry(key).or_default();
        match plane {
            Plane::Value => cell.value = Some(value),
            Plane::Weight => cell.weight = Some(value),
        }
        Ok(())
    }

    fn exact_value(&self, key: &CellKey) -> Option<f64> {
        self.cells.get(key)?.value.as_ref()?.as_f64()
    }

    pub fn writer<'a>(&'a mut self, floor: &str) -> Option<Writer<'a>> {
        let floor = self.layer_index(floor)?;
        Some(Writer {
            tensor: self,
            floor,
        })
    }

    /// Remove one exact preference set from a layer.
    pub fn clear_scope(&mut self, layer: &str, scope: &Scope) -> Result<(), WriteError> {
        let scope = self.normalize_scope(layer, scope)?;
        self.cells
            .retain(|key, _| key.layer != layer || key.scope != scope);
        Ok(())
    }

    fn matches(scope: &[(String, String)], cursor: &Cursor) -> bool {
        scope
            .iter()
            .all(|(axis, value)| cursor.get(axis).is_some_and(|current| current == value))
    }

    fn specificity_cmp(&self, left: &CellKey, right: &CellKey) -> Ordering {
        let left_layer = self.layer_index(&left.layer).unwrap_or_default();
        let right_layer = self.layer_index(&right.layer).unwrap_or_default();
        left_layer.cmp(&right_layer).then_with(|| {
            // At one layer, a preference pinned on the finest matching axis wins;
            // then the next-finest, and so on. This makes a source-specific cell
            // outrank a session-only cell inside the source layer.
            for axis in self.layers.iter().rev() {
                let left_has = left.scope.iter().any(|(candidate, _)| candidate == axis);
                let right_has = right.scope.iter().any(|(candidate, _)| candidate == axis);
                match left_has.cmp(&right_has) {
                    Ordering::Equal => {}
                    ordering => return ordering,
                }
            }
            // Free context facets are not preference owners. More matching
            // conditions are nevertheless more specific inside one owner layer.
            left.scope
                .len()
                .cmp(&right.scope.len())
                .then_with(|| left.scope.cmp(&right.scope))
        })
    }

    fn best<'a>(
        &'a self,
        cursor: &Cursor,
        option: &str,
        dim: &str,
        plane: Plane,
    ) -> Option<(&'a CellKey, &'a Value)> {
        self.cells
            .iter()
            .filter(|(key, _)| {
                key.option == option && key.dim == dim && Self::matches(&key.scope, cursor)
            })
            .filter_map(|(key, cell)| {
                let value = match plane {
                    Plane::Value => cell.value.as_ref(),
                    Plane::Weight => cell.weight.as_ref(),
                }?;
                Some((key, value))
            })
            .max_by(|(left, _), (right, _)| self.specificity_cmp(left, right))
    }

    fn origin(key: &CellKey) -> Origin {
        Origin {
            layer: key.layer.clone(),
            scope: Self::scope_map(&key.scope),
        }
    }

    pub fn value(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<f64> {
        self.best(cursor, option, dim, Plane::Value)
            .and_then(|(_, value)| value.as_f64())
    }

    pub fn bytes(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<&[u8]> {
        self.best(cursor, option, dim, Plane::Value)
            .and_then(|(_, value)| value.as_bytes())
    }

    pub fn weight(&self, cursor: &Cursor, option: &str, dim: &str) -> Option<f64> {
        self.best(cursor, option, dim, Plane::Weight)
            .or_else(|| self.best(cursor, SHARED, dim, Plane::Weight))
            .and_then(|(_, value)| value.as_f64())
    }

    pub fn options(&self, cursor: &Cursor) -> BTreeSet<String> {
        self.cells
            .keys()
            .filter(|key| key.option != SHARED && Self::matches(&key.scope, cursor))
            .map(|key| key.option.clone())
            .collect()
    }

    /// Resolve and retain the contributing layer/scope origins for observation.
    pub fn resolve_explained(&self, cursor: &Cursor) -> Vec<ResolvedOption> {
        // Fold every compatible cell once. The previous implementation called
        // `best` repeatedly for every option/dimension pair, which made one
        // decision quadratic in a large sparse participant tensor.
        let mut dimensions_by_option: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut values: BTreeMap<(String, String), (&CellKey, &Value)> = BTreeMap::new();
        let mut weights: BTreeMap<(String, String), (&CellKey, &Value)> = BTreeMap::new();
        for (key, cell) in &self.cells {
            if !Self::matches(&key.scope, cursor) {
                continue;
            }
            if key.option != SHARED {
                dimensions_by_option
                    .entry(key.option.clone())
                    .or_default()
                    .insert(key.dim.clone());
            }
            if let Some(value) = cell.value.as_ref() {
                let slot = values
                    .entry((key.option.clone(), key.dim.clone()))
                    .or_insert((key, value));
                if self.specificity_cmp(key, slot.0).is_gt() {
                    *slot = (key, value);
                }
            }
            if let Some(weight) = cell.weight.as_ref() {
                let slot = weights
                    .entry((key.option.clone(), key.dim.clone()))
                    .or_insert((key, weight));
                if self.specificity_cmp(key, slot.0).is_gt() {
                    *slot = (key, weight);
                }
            }
        }

        let mut resolved = Vec::new();
        for (option, option_dimensions) in dimensions_by_option {
            let mut score = 0.0;
            let mut viable = true;
            let mut routable = false;
            let mut dimensions = Vec::new();
            for dimension in option_dimensions {
                let Some((value_key, value)) = values.get(&(option.clone(), dimension.clone()))
                else {
                    continue;
                };
                let Some(value) = value.as_f64() else {
                    continue;
                };
                routable = true;
                let weight = weights
                    .get(&(option.clone(), dimension.clone()))
                    .or_else(|| weights.get(&(SHARED.to_owned(), dimension.clone())));
                let scalar_weight = weight.and_then(|(_, value)| value.as_f64());
                let contribution = match scalar_weight {
                    Some(weight) if weight.is_finite() && value.is_finite() => value * weight,
                    _ => 0.0,
                };
                if !value.is_finite() {
                    viable = false;
                }
                score += contribution;
                dimensions.push(DimensionTrace {
                    dimension,
                    value,
                    weight: scalar_weight,
                    contribution,
                    value_origin: Self::origin(value_key),
                    weight_origin: weight.map(|(key, _)| Self::origin(key)),
                });
            }
            if routable {
                resolved.push(ResolvedOption {
                    option,
                    viable,
                    score,
                    dimensions,
                });
            }
        }
        resolved.sort_by(|left, right| {
            right
                .viable
                .cmp(&left.viable)
                .then_with(|| fcmp(left.score, right.score))
                .then_with(|| left.option.cmp(&right.option))
        });
        resolved
    }

    pub fn resolve(&self, cursor: &Cursor) -> Vec<String> {
        self.resolve_explained(cursor)
            .into_iter()
            .filter(|candidate| candidate.viable)
            .map(|candidate| candidate.option)
            .collect()
    }

    fn is_descendant_scope(key_scope: &[(String, String)], ancestor: &Scope) -> bool {
        ancestor
            .iter()
            .filter(|(_, value)| !value.is_empty())
            .all(|(axis, value)| {
                key_scope
                    .iter()
                    .any(|(candidate, current)| candidate == axis && current == value)
            })
    }

    fn descendant_values(
        &self,
        from_layer: &str,
        ancestor: &Scope,
        option: &str,
        dim: &str,
    ) -> Vec<f64> {
        self.cells
            .iter()
            .filter(|(key, _)| {
                key.layer == from_layer
                    && key.option == option
                    && key.dim == dim
                    && Self::is_descendant_scope(&key.scope, ancestor)
            })
            .filter_map(|(_, cell)| cell.value.as_ref()?.as_f64())
            .filter(|value| value.is_finite())
            .collect()
    }

    fn descendant_pairs(&self, from_layer: &str, ancestor: &Scope) -> BTreeSet<(String, String)> {
        self.cells
            .keys()
            .filter(|key| {
                key.layer == from_layer
                    && key.option != SHARED
                    && Self::is_descendant_scope(&key.scope, ancestor)
            })
            .map(|key| (key.option.clone(), key.dim.clone()))
            .collect()
    }

    fn check_reduction(
        &self,
        from_layer: &str,
        to_layer: &str,
        scope: &Scope,
    ) -> Result<(), WriteError> {
        let from = self
            .layer_index(from_layer)
            .ok_or(WriteError::UnknownLayer)?;
        let to = self.layer_index(to_layer).ok_or(WriteError::UnknownLayer)?;
        if from <= to {
            return Err(WriteError::WriteUp);
        }
        self.normalize_scope(to_layer, scope)?;
        Ok(())
    }

    /// Median-project descendant preferences into one ancestor preference set.
    pub fn reduce_median(
        &mut self,
        from_layer: &str,
        to_layer: &str,
        target_scope: &Scope,
        option: &str,
        dim: &str,
    ) -> Result<Option<PreferenceChange>, WriteError> {
        self.check_reduction(from_layer, to_layer, target_scope)?;
        let mut values = self.descendant_values(from_layer, target_scope, option, dim);
        if values.is_empty() {
            return Ok(None);
        }
        values.sort_by(|left, right| fcmp(*left, *right));
        let middle = values.len() / 2;
        let median = if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        };
        self.replace_reduced_value(to_layer, target_scope, option, dim, median)
            .map(Some)
    }

    /// Median-project every descendant `(option, dimension)` pair into one
    /// ancestor preference set, in canonical pair order.
    pub fn reduce_all_median(
        &mut self,
        from_layer: &str,
        to_layer: &str,
        target_scope: &Scope,
    ) -> Result<Vec<PreferenceChange>, WriteError> {
        self.check_reduction(from_layer, to_layer, target_scope)?;
        let pairs = self.descendant_pairs(from_layer, target_scope);
        let mut changes = Vec::new();
        for (option, dim) in pairs {
            if let Some(change) =
                self.reduce_median(from_layer, to_layer, target_scope, &option, &dim)?
            {
                changes.push(change);
            }
        }
        Ok(changes)
    }

    /// Sum-project descendant preferences into one ancestor preference set.
    pub fn reduce_sum(
        &mut self,
        from_layer: &str,
        to_layer: &str,
        target_scope: &Scope,
        option: &str,
        dim: &str,
    ) -> Result<Option<PreferenceChange>, WriteError> {
        self.check_reduction(from_layer, to_layer, target_scope)?;
        let mut values = self.descendant_values(from_layer, target_scope, option, dim);
        if values.is_empty() {
            return Ok(None);
        }
        values.sort_by(|left, right| fcmp(*left, *right));
        self.replace_reduced_value(
            to_layer,
            target_scope,
            option,
            dim,
            values.into_iter().sum(),
        )
        .map(Some)
    }

    fn replace_reduced_value(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        next: f64,
    ) -> Result<PreferenceChange, WriteError> {
        let key = self.key(layer, scope, option, dim)?;
        let previous = self.exact_value(&key);
        self.put(layer, scope, option, dim, Plane::Value, Value::F64(next))?;
        Ok(PreferenceChange {
            layer: layer.to_owned(),
            scope: scope.clone(),
            option: option.to_owned(),
            dimension: dim.to_owned(),
            previous,
            next,
        })
    }

    pub fn support(&self, from_layer: &str, ancestor: &Scope, option: &str, dim: &str) -> usize {
        self.descendant_values(from_layer, ancestor, option, dim)
            .len()
    }

    /// Corroborate positive, independent descendant attestations into an ancestor.
    pub fn corroborate(
        &mut self,
        from_layer: &str,
        to_layer: &str,
        target_scope: &Scope,
        subject: (&str, &str),
        reporter_axis: &str,
        quorum: usize,
    ) -> Result<Option<PreferenceChange>, WriteError> {
        let (option, dim) = subject;
        self.check_reduction(from_layer, to_layer, target_scope)?;
        let reporters: BTreeSet<String> = self
            .cells
            .iter()
            .filter(|(key, cell)| {
                key.layer == from_layer
                    && key.option == option
                    && key.dim == dim
                    && Self::is_descendant_scope(&key.scope, target_scope)
                    && cell
                        .value
                        .as_ref()
                        .and_then(Value::as_f64)
                        .is_some_and(|value| value > 0.0)
            })
            .filter_map(|(key, _)| {
                key.scope
                    .iter()
                    .find(|(axis, _)| axis == reporter_axis)
                    .map(|(_, value)| value.clone())
            })
            .filter(|reporter| reporter != option)
            .collect();
        if reporters.len() < quorum {
            return Ok(None);
        }
        self.replace_reduced_value(to_layer, target_scope, option, dim, 1.0)
            .map(Some)
    }

    /// Corroborate every descendant `(option, dimension)` pair into one
    /// ancestor preference set, in canonical pair order.
    pub fn corroborate_all(
        &mut self,
        from_layer: &str,
        to_layer: &str,
        target_scope: &Scope,
        reporter_axis: &str,
        quorum: usize,
    ) -> Result<Vec<PreferenceChange>, WriteError> {
        self.check_reduction(from_layer, to_layer, target_scope)?;
        let pairs = self.descendant_pairs(from_layer, target_scope);
        let mut changes = Vec::new();
        for (option, dim) in pairs {
            if let Some(change) = self.corroborate(
                from_layer,
                to_layer,
                target_scope,
                (&option, &dim),
                reporter_axis,
                quorum,
            )? {
                changes.push(change);
            }
        }
        Ok(changes)
    }

    pub fn cells(&self) -> Vec<JsonCell> {
        self.cells
            .iter()
            .map(|(key, cell)| JsonCell {
                layer: key.layer.clone(),
                scope: Self::scope_map(&key.scope),
                option: key.option.clone(),
                dim: key.dim.clone(),
                v: cell.value.clone(),
                w: cell.weight.clone(),
            })
            .collect()
    }

    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            layers: &'a [String],
            cells: Vec<JsonCell>,
        }
        serde_json::to_string(&Snapshot {
            layers: &self.layers,
            cells: self.cells(),
        })
        .unwrap_or_else(|_| r#"{"layers":[],"cells":[]}"#.to_owned())
    }

    pub fn apply_json(&mut self, cells_json: &str) {
        let cells: Vec<JsonCell> = serde_json::from_str(cells_json).unwrap_or_default();
        for cell in cells {
            if let Some(value) = cell.v {
                let _ = self.put(
                    &cell.layer,
                    &cell.scope,
                    &cell.option,
                    &cell.dim,
                    Plane::Value,
                    value,
                );
            }
            if let Some(weight) = cell.w {
                let _ = self.put(
                    &cell.layer,
                    &cell.scope,
                    &cell.option,
                    &cell.dim,
                    Plane::Weight,
                    weight,
                );
            }
        }
    }

    pub fn from_json(snapshot: &str) -> Option<Self> {
        #[derive(Deserialize)]
        struct Snapshot {
            layers: Vec<String>,
            cells: Vec<JsonCell>,
        }
        let snapshot: Snapshot = serde_json::from_str(snapshot).ok()?;
        let mut tensor = Tensor::new(snapshot.layers);
        let cells = serde_json::to_string(&snapshot.cells).ok()?;
        tensor.apply_json(&cells);
        Some(tensor)
    }
}

/// A structurally attenuated capability: writes at its floor or finer, never up.
pub struct Writer<'a> {
    tensor: &'a mut Tensor,
    floor: usize,
}

impl Writer<'_> {
    fn check(&self, layer: &str, scope: &Scope) -> Result<(), WriteError> {
        let target = self
            .tensor
            .layer_index(layer)
            .ok_or(WriteError::UnknownLayer)?;
        if target < self.floor {
            return Err(WriteError::WriteUp);
        }
        self.tensor.normalize_scope(layer, scope)?;
        Ok(())
    }

    fn write(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        plane: Plane,
        value: Value,
    ) -> Result<(), WriteError> {
        self.check(layer, scope)?;
        self.tensor.put(layer, scope, option, dim, plane, value)
    }

    pub fn set_value(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        value: f64,
    ) -> Result<(), WriteError> {
        self.write(layer, scope, option, dim, Plane::Value, Value::F64(value))
    }

    pub fn set_bytes(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        value: Vec<u8>,
    ) -> Result<(), WriteError> {
        self.write(layer, scope, option, dim, Plane::Value, Value::Bytes(value))
    }

    pub fn set_weight(
        &mut self,
        layer: &str,
        scope: &Scope,
        dim: &str,
        weight: f64,
    ) -> Result<(), WriteError> {
        self.write(layer, scope, SHARED, dim, Plane::Weight, Value::F64(weight))
    }

    pub fn set_option_weight(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        weight: f64,
    ) -> Result<(), WriteError> {
        self.write(layer, scope, option, dim, Plane::Weight, Value::F64(weight))
    }

    /// Revise one exact layer-local preference by an EWMA step.
    ///
    /// The product chooses the target layer and turns its outcome into a scalar;
    /// Sporewright guarantees that no other layer is silently mutated.
    pub fn nudge_value(
        &mut self,
        layer: &str,
        scope: &Scope,
        option: &str,
        dim: &str,
        observation: f64,
        rate: f64,
    ) -> Result<PreferenceChange, WriteError> {
        self.check(layer, scope)?;
        let key = self.tensor.key(layer, scope, option, dim)?;
        let previous = self.tensor.exact_value(&key);
        let rate = if rate.is_finite() {
            rate.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let next = match previous {
            Some(previous) if previous.is_finite() && observation.is_finite() => {
                previous + rate * (observation - previous)
            }
            _ => observation,
        };
        self.tensor
            .put(layer, scope, option, dim, Plane::Value, Value::F64(next))?;
        Ok(PreferenceChange {
            layer: layer.to_owned(),
            scope: scope.clone(),
            option: option.to_owned(),
            dimension: dim.to_owned(),
            previous,
            next,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_scope_keeps_device_feedback_source_local() {
        let mut tensor = Tensor::new(["session", "source", "job", "device"]);
        {
            let mut writer = tensor.writer("session").unwrap();
            writer
                .set_weight("session", &Scope::new(), "reliability", 1.0)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "browser", "reliability", 0.1)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "native", "reliability", 0.2)
                .unwrap();
        }
        tensor
            .writer("device")
            .unwrap()
            .set_value(
                "device",
                &scope([("source", "nzz"), ("device", "d1")]),
                "browser",
                "reliability",
                1.0,
            )
            .unwrap();

        assert_eq!(
            tensor.resolve(&scope([("source", "nzz"), ("device", "d1")])),
            vec!["native", "browser"]
        );
        assert_eq!(
            tensor.resolve(&scope([("source", "proton"), ("device", "d1")])),
            vec!["browser", "native"]
        );
        assert_eq!(
            tensor.resolve(&scope([("source", "nzz"), ("device", "d2")])),
            vec!["browser", "native"]
        );
    }

    #[test]
    fn scoped_reduction_does_not_mix_sources() {
        let mut tensor = Tensor::new(["session", "source", "job", "device"]);
        for (source_id, device, value) in [
            ("nzz", "a", 1.0),
            ("nzz", "b", 0.8),
            ("proton", "a", 0.0),
            ("proton", "b", 0.2),
        ] {
            tensor
                .writer("device")
                .unwrap()
                .set_value(
                    "device",
                    &scope([("source", source_id), ("device", device)]),
                    "browser",
                    "reliability",
                    value,
                )
                .unwrap();
        }
        tensor
            .reduce_median(
                "device",
                "source",
                &scope([("source", "nzz")]),
                "browser",
                "reliability",
            )
            .unwrap();
        tensor
            .reduce_median(
                "device",
                "source",
                &scope([("source", "proton")]),
                "browser",
                "reliability",
            )
            .unwrap();

        assert_eq!(
            tensor.value(&scope([("source", "nzz")]), "browser", "reliability"),
            Some(0.9)
        );
        assert_eq!(
            tensor.value(&scope([("source", "proton")]), "browser", "reliability"),
            Some(0.1)
        );
    }

    #[test]
    fn layer_local_nudge_changes_only_its_context() {
        let mut tensor = Tensor::new(["workspace", "stage", "consumer", "instance", "device"]);
        {
            let mut writer = tensor.writer("workspace").unwrap();
            writer
                .set_weight("workspace", &Scope::new(), "quality", 1.0)
                .unwrap();
            writer
                .set_value("workspace", &Scope::new(), "groq", "quality", 0.2)
                .unwrap();
            writer
                .set_value("workspace", &Scope::new(), "anthropic", "quality", 0.3)
                .unwrap();
        }
        tensor
            .writer("consumer")
            .unwrap()
            .nudge_value(
                "consumer",
                &scope([("stage", "evaluate"), ("consumer", "fit")]),
                "groq",
                "quality",
                1.0,
                1.0,
            )
            .unwrap();

        assert_eq!(
            tensor.resolve(&scope([("stage", "evaluate"), ("consumer", "fit")])),
            vec!["anthropic", "groq"]
        );
        assert_eq!(
            tensor.resolve(&scope([("stage", "evaluate"), ("consumer", "salary")])),
            vec!["groq", "anthropic"]
        );
    }

    #[test]
    fn explanations_name_the_preference_layer() {
        let mut tensor = Tensor::new(["session", "source", "job", "device"]);
        let source = scope([("source", "nzz")]);
        {
            let mut writer = tensor.writer("session").unwrap();
            writer
                .set_weight("session", &Scope::new(), "cost", 1.0)
                .unwrap();
            writer
                .set_value("session", &Scope::new(), "browser", "cost", 0.0)
                .unwrap();
            writer
                .set_value("source", &source, "browser", "cost", 2.0)
                .unwrap();
        }
        let explained = tensor.resolve_explained(&source);
        assert_eq!(explained[0].dimensions[0].value_origin.layer, "source");
        assert_eq!(explained[0].dimensions[0].value_origin.scope, source);
        assert_eq!(
            explained[0].dimensions[0]
                .weight_origin
                .as_ref()
                .unwrap()
                .layer,
            "session"
        );
    }

    #[test]
    fn finer_scope_cannot_be_written_into_coarser_layer() {
        let mut tensor = Tensor::new(["session", "source", "job", "device"]);
        let error = tensor.writer("session").unwrap().set_value(
            "source",
            &scope([("device", "d1")]),
            "browser",
            "cost",
            1.0,
        );
        assert_eq!(error, Err(WriteError::ScopeOutsideLayer));
    }

    #[test]
    fn context_facets_are_independent_from_preference_ownership() {
        let mut tensor = Tensor::new(["workspace", "stage", "consumer"]);
        let mut writer = tensor.writer("workspace").unwrap();
        writer
            .set_value("workspace", &Scope::new(), "local", "cost", 0.2)
            .unwrap();
        writer
            .set_value(
                "workspace",
                &scope([("capability", "chat")]),
                "local",
                "cost",
                0.8,
            )
            .unwrap();

        assert_eq!(
            tensor.value(&scope([("capability", "chat")]), "local", "cost"),
            Some(0.8)
        );
        assert_eq!(
            tensor.value(&scope([("capability", "embedding")]), "local", "cost"),
            Some(0.2)
        );
    }

    #[test]
    fn snapshot_round_trip_preserves_scoped_preferences() {
        let mut tensor = Tensor::new(["session", "source", "device"]);
        let source = scope([("source", "nzz")]);
        tensor
            .writer("source")
            .unwrap()
            .set_value("source", &source, "browser", "latency", 0.5)
            .unwrap();
        let restored = Tensor::from_json(&tensor.to_json()).unwrap();
        assert_eq!(restored, tensor);
    }
}
