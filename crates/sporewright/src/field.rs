// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Sparse addressed residual fields with cached Gaussian message passing.
//!
//! This module is the v2 mathematical core. An address is an exact path through
//! product-declared layers. Numeric priors add along that path. Observations are
//! retained as Gaussian natural sufficient statistics and update cached subtree
//! messages on the leaf-to-root path only.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// The shared option slot used by dimension weights.
pub const SHARED: &str = "";
const MAX_SAFE_OBSERVATIONS: u64 = 9_007_199_254_740_991;

/// A named path through a product-declared address schema.
pub type Address = BTreeMap<String, String>;

/// Build an address while omitting empty coordinates.
pub fn address<const N: usize>(coordinates: [(&str, &str); N]) -> Address {
    coordinates
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(layer, value)| (layer.to_owned(), value.to_owned()))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldError {
    UnknownLayer,
    UnknownCoordinate,
    AddressGap,
    AddressTooShallow,
    WriteUp,
    InvalidVariance,
    InvalidObservation,
    InvalidDiscount,
    NoEvidence,
    RootCompaction,
    FrozenProcessVariance,
    InvalidSchema,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Natural {
    pub precision: f64,
    pub information: f64,
}

impl Natural {
    fn add(self, other: Self) -> Self {
        Self {
            precision: self.precision + other.precision,
            information: self.information + other.information,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            precision: (self.precision - other.precision).max(0.0),
            information: self.information - other.information,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub precision: f64,
    pub information: f64,
    pub observations: u64,
}

impl Evidence {
    pub fn from_observation(value: f64, variance: f64) -> Result<Self, FieldError> {
        if !value.is_finite() {
            return Err(FieldError::InvalidObservation);
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FieldError::InvalidVariance);
        }
        let precision = variance.recip();
        Ok(Self {
            precision,
            information: value * precision,
            observations: 1,
        })
    }

    fn natural(self) -> Natural {
        Natural {
            precision: self.precision,
            information: self.information,
        }
    }

    fn merge(&mut self, other: Self) {
        self.precision += other.precision;
        self.information += other.information;
        self.observations = self
            .observations
            .saturating_add(other.observations)
            .min(MAX_SAFE_OBSERVATIONS);
    }

    fn discount(&mut self, factor: f64) {
        self.precision *= factor;
        self.information *= factor;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Cell {
    prior: Option<f64>,
    weight: Option<f64>,
    gate: Option<bool>,
    evidence: Evidence,
    child_messages: Natural,
    message_to_parent: Natural,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    path: Vec<String>,
    option: String,
    dimension: String,
}

/// One serialized sparse field cell. Cached messages are deliberately omitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonFieldCell {
    pub address: Address,
    pub option: String,
    pub dimension: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<bool>,
    #[serde(default, skip_serializing_if = "is_empty_evidence")]
    pub evidence: Evidence,
}

fn is_empty_evidence(value: &Evidence) -> bool {
    value.observations == 0 && value.precision == 0.0 && value.information == 0.0
}

fn default_root_process_variance() -> f64 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualTrace {
    pub address: Address,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub residual_mean: f64,
    pub declared_prior: f64,
    pub observations: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDimensionTrace {
    pub dimension: String,
    pub mean: f64,
    pub variance: f64,
    pub weight: f64,
    pub contribution: f64,
    pub contributions: Vec<ResidualTrace>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldResolvedOption {
    pub option: String,
    pub viable: bool,
    pub expected_score: f64,
    pub score_variance: f64,
    pub decision_score: f64,
    pub dimensions: Vec<FieldDimensionTrace>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionPolicy {
    /// Zero exploits the posterior mean. Positive values reward uncertainty.
    pub temperature: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolveWork {
    pub address_nodes: usize,
    pub stored_cells_visited: usize,
    pub parameter_lookups: usize,
    pub ranked_options: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDecision {
    pub address: Address,
    pub policy: DecisionPolicy,
    pub alternatives: Vec<FieldResolvedOption>,
    pub work: ResolveWork,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutingReceipt {
    pub tensor_revision: String,
    pub decision: FieldDecision,
    pub selected: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedbackTrace {
    pub address: Address,
    pub option: String,
    pub dimension: String,
    pub observed: f64,
    pub observation_variance: f64,
    pub updated_nodes: usize,
    pub posterior_before: Option<(f64, f64)>,
    pub posterior_after: (f64, f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscountTrace {
    pub address: Address,
    pub option: String,
    pub dimension: String,
    pub factor: f64,
    pub precision_before: f64,
    pub precision_after: f64,
    pub updated_nodes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionTrace {
    pub address: Address,
    pub compacted_into: Address,
    pub option: String,
    pub dimension: String,
    pub observations: u64,
    pub removed_cells: usize,
    pub updated_nodes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Gaussian {
    mean: f64,
    variance: f64,
}

fn gaussian_from_natural(natural: Natural) -> Option<Gaussian> {
    if natural.precision <= 0.0 || !natural.precision.is_finite() {
        return None;
    }
    Some(Gaussian {
        mean: natural.information / natural.precision,
        variance: natural.precision.recip(),
    })
}

fn natural_from_gaussian(gaussian: Gaussian) -> Natural {
    let precision = gaussian.variance.recip();
    Natural {
        precision,
        information: gaussian.mean * precision,
    }
}

fn combine(prior: Gaussian, likelihood: Natural) -> Gaussian {
    let combined = natural_from_gaussian(prior).add(likelihood);
    gaussian_from_natural(combined).unwrap_or(prior)
}

fn fcmp(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right)
        .unwrap_or(match (left.is_nan(), right.is_nan()) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => Ordering::Equal,
        })
}

/// A sparse addressed field. Stored state grows only when a product declares a
/// parameter or supplies evidence for an address/option/dimension coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct AddressedField {
    pub layers: Vec<String>,
    default_root_process_variance: f64,
    default_process_variance: Vec<f64>,
    dimension_root_process_variance: BTreeMap<String, f64>,
    dimension_process_variance: BTreeMap<String, Vec<f64>>,
    cells: BTreeMap<CellKey, Cell>,
    state_dimensions: BTreeMap<(Vec<String>, String), BTreeSet<String>>,
    path_index: BTreeMap<Vec<String>, BTreeSet<(String, String)>>,
    process_variance_frozen: bool,
}

impl AddressedField {
    pub fn try_new(
        layers: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, FieldError> {
        let layers: Vec<String> = layers.into_iter().map(Into::into).collect();
        let unique: BTreeSet<_> = layers.iter().collect();
        if layers.iter().any(|layer| layer.is_empty()) || unique.len() != layers.len() {
            return Err(FieldError::InvalidSchema);
        }
        Ok(Self {
            default_root_process_variance: 1.0,
            default_process_variance: vec![1.0; layers.len()],
            layers,
            dimension_root_process_variance: BTreeMap::new(),
            dimension_process_variance: BTreeMap::new(),
            cells: BTreeMap::new(),
            state_dimensions: BTreeMap::new(),
            path_index: BTreeMap::new(),
            process_variance_frozen: false,
        })
    }

    /// Construct a field from a static product schema.
    ///
    /// # Panics
    ///
    /// Panics if the schema contains empty or duplicate layer names.
    /// Use [`AddressedField::try_new`] for runtime-supplied schemas.
    pub fn new(layers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::try_new(layers).expect("address schemas must contain unique non-empty layers")
    }

    fn layer_index(&self, layer: &str) -> Option<usize> {
        self.layers.iter().position(|candidate| candidate == layer)
    }

    fn path(&self, input: &Address) -> Result<Vec<String>, FieldError> {
        if input.keys().any(|axis| self.layer_index(axis).is_none()) {
            return Err(FieldError::UnknownCoordinate);
        }
        let mut path = Vec::new();
        let mut gap = false;
        for layer in &self.layers {
            match input.get(layer).filter(|value| !value.is_empty()) {
                Some(value) if !gap => path.push(value.clone()),
                Some(_) => return Err(FieldError::AddressGap),
                None => gap = true,
            }
        }
        Ok(path)
    }

    fn address_for_path(&self, path: &[String]) -> Address {
        self.layers
            .iter()
            .zip(path)
            .map(|(layer, value)| (layer.clone(), value.clone()))
            .collect()
    }

    fn key(path: &[String], option: &str, dimension: &str) -> CellKey {
        CellKey {
            path: path.to_vec(),
            option: option.to_owned(),
            dimension: dimension.to_owned(),
        }
    }

    fn process_variance(&self, dimension: &str, depth: usize) -> f64 {
        self.dimension_process_variance
            .get(dimension)
            .unwrap_or(&self.default_process_variance)[depth]
    }

    fn root_process_variance(&self, dimension: &str) -> f64 {
        self.dimension_root_process_variance
            .get(dimension)
            .copied()
            .unwrap_or(self.default_root_process_variance)
    }

    pub fn set_default_root_process_variance(&mut self, variance: f64) -> Result<(), FieldError> {
        if self.process_variance_frozen {
            return Err(FieldError::FrozenProcessVariance);
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FieldError::InvalidVariance);
        }
        self.default_root_process_variance = variance;
        self.rebuild_messages();
        Ok(())
    }

    pub fn set_dimension_root_process_variance(
        &mut self,
        dimension: &str,
        variance: f64,
    ) -> Result<(), FieldError> {
        if self.process_variance_frozen {
            return Err(FieldError::FrozenProcessVariance);
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FieldError::InvalidVariance);
        }
        self.dimension_root_process_variance
            .insert(dimension.to_owned(), variance);
        self.rebuild_messages();
        Ok(())
    }

    pub fn set_default_process_variance(
        &mut self,
        layer: &str,
        variance: f64,
    ) -> Result<(), FieldError> {
        if self.process_variance_frozen {
            return Err(FieldError::FrozenProcessVariance);
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FieldError::InvalidVariance);
        }
        let index = self.layer_index(layer).ok_or(FieldError::UnknownLayer)?;
        self.default_process_variance[index] = variance;
        self.rebuild_messages();
        Ok(())
    }

    pub fn set_dimension_process_variance(
        &mut self,
        dimension: &str,
        layer: &str,
        variance: f64,
    ) -> Result<(), FieldError> {
        if self.process_variance_frozen {
            return Err(FieldError::FrozenProcessVariance);
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(FieldError::InvalidVariance);
        }
        let index = self.layer_index(layer).ok_or(FieldError::UnknownLayer)?;
        let variances = self
            .dimension_process_variance
            .entry(dimension.to_owned())
            .or_insert_with(|| self.default_process_variance.clone());
        variances[index] = variance;
        self.rebuild_messages();
        Ok(())
    }

    pub fn writer(&mut self, floor: &str) -> Option<FieldWriter<'_>> {
        let floor = self.layer_index(floor)?;
        Some(FieldWriter {
            field: self,
            floor: Some(floor),
        })
    }

    /// A product-authority writer that can declare policy at the implicit root.
    pub fn root_writer(&mut self) -> FieldWriter<'_> {
        FieldWriter {
            field: self,
            floor: None,
        }
    }

    fn ensure_path(&mut self, path: &[String], option: &str, dimension: &str) {
        for depth in 0..=path.len() {
            let prefix = &path[..depth];
            let key = Self::key(prefix, option, dimension);
            self.cells.entry(key).or_default();
            self.state_dimensions
                .entry((prefix.to_vec(), option.to_owned()))
                .or_default()
                .insert(dimension.to_owned());
        }
    }

    fn index_coordinate(&mut self, path: &[String], option: &str, dimension: &str) {
        self.path_index
            .entry(path.to_vec())
            .or_default()
            .insert((option.to_owned(), dimension.to_owned()));
    }

    fn subtree(cell: &Cell) -> Natural {
        cell.evidence.natural().add(cell.child_messages)
    }

    fn message_for(&self, key: &CellKey) -> Natural {
        if key.path.is_empty() {
            return Natural::default();
        }
        let Some(cell) = self.cells.get(key) else {
            return Natural::default();
        };
        let likelihood = Self::subtree(cell);
        let Some(like) = gaussian_from_natural(likelihood) else {
            return Natural::default();
        };
        let depth = key.path.len() - 1;
        let transition = self.process_variance(&key.dimension, depth);
        let prior = cell.prior.unwrap_or(0.0);
        natural_from_gaussian(Gaussian {
            mean: like.mean - prior,
            variance: like.variance + transition,
        })
    }

    fn recompute_up(&mut self, path: &[String], option: &str, dimension: &str) -> usize {
        let mut updated = 0;
        for depth in (0..=path.len()).rev() {
            let prefix = &path[..depth];
            let key = Self::key(prefix, option, dimension);
            let previous = self
                .cells
                .get(&key)
                .map(|cell| cell.message_to_parent)
                .unwrap_or_default();
            let next = self.message_for(&key);
            if let Some(cell) = self.cells.get_mut(&key) {
                cell.message_to_parent = next;
            }
            updated += 1;
            if depth > 0 {
                let parent_key = Self::key(&path[..depth - 1], option, dimension);
                if let Some(parent) = self.cells.get_mut(&parent_key) {
                    parent.child_messages = parent.child_messages.sub(previous).add(next);
                }
            }
        }
        updated
    }

    fn rebuild_messages(&mut self) {
        for cell in self.cells.values_mut() {
            cell.child_messages = Natural::default();
            cell.message_to_parent = Natural::default();
        }
        let keys: Vec<CellKey> = self.cells.keys().cloned().collect();
        let mut keys = keys;
        keys.sort_by(|left, right| right.path.len().cmp(&left.path.len()).then(left.cmp(right)));
        for key in keys {
            let next = self.message_for(&key);
            if let Some(cell) = self.cells.get_mut(&key) {
                cell.message_to_parent = next;
            }
            if !key.path.is_empty() {
                let parent_key =
                    Self::key(&key.path[..key.path.len() - 1], &key.option, &key.dimension);
                if let Some(parent) = self.cells.get_mut(&parent_key) {
                    parent.child_messages = parent.child_messages.add(next);
                }
            }
        }
    }

    fn set_prior(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
        value: f64,
    ) -> Result<(), FieldError> {
        if !value.is_finite() {
            return Err(FieldError::InvalidObservation);
        }
        self.ensure_path(path, option, dimension);
        self.index_coordinate(path, option, dimension);
        let key = Self::key(path, option, dimension);
        self.cells.get_mut(&key).expect("ensured cell").prior = Some(value);
        self.recompute_up(path, option, dimension);
        Ok(())
    }

    fn set_weight(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
        value: f64,
    ) -> Result<(), FieldError> {
        if !value.is_finite() {
            return Err(FieldError::InvalidObservation);
        }
        self.ensure_path(path, option, dimension);
        self.index_coordinate(path, option, dimension);
        let key = Self::key(path, option, dimension);
        self.cells.get_mut(&key).expect("ensured cell").weight = Some(value);
        Ok(())
    }

    fn set_gate(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
        value: bool,
    ) -> Result<(), FieldError> {
        self.ensure_path(path, option, dimension);
        self.index_coordinate(path, option, dimension);
        let key = Self::key(path, option, dimension);
        self.cells.get_mut(&key).expect("ensured cell").gate = Some(value);
        Ok(())
    }

    fn merge_evidence(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
        evidence: Evidence,
    ) -> Result<usize, FieldError> {
        if !evidence.precision.is_finite()
            || evidence.precision < 0.0
            || !evidence.information.is_finite()
            || (evidence.precision == 0.0 && evidence.information != 0.0)
            || evidence.observations > MAX_SAFE_OBSERVATIONS
        {
            return Err(FieldError::InvalidObservation);
        }
        let key = Self::key(path, option, dimension);
        let target = self
            .cells
            .get(&key)
            .map(|cell| cell.evidence)
            .unwrap_or_default();
        if !(target.precision + evidence.precision).is_finite()
            || !(target.information + evidence.information).is_finite()
        {
            return Err(FieldError::InvalidObservation);
        }
        self.ensure_path(path, option, dimension);
        self.index_coordinate(path, option, dimension);
        self.cells
            .get_mut(&key)
            .expect("ensured cell")
            .evidence
            .merge(evidence);
        Ok(self.recompute_up(path, option, dimension))
    }

    fn discount_evidence(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
        factor: f64,
    ) -> Result<DiscountTrace, FieldError> {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(FieldError::InvalidDiscount);
        }
        let key = Self::key(path, option, dimension);
        let precision_before = self
            .cells
            .get(&key)
            .filter(|cell| cell.evidence.observations > 0)
            .map(|cell| cell.evidence.precision)
            .ok_or(FieldError::NoEvidence)?;
        self.cells
            .get_mut(&key)
            .expect("checked evidence cell")
            .evidence
            .discount(factor);
        let precision_after = self
            .cells
            .get(&key)
            .expect("checked evidence cell")
            .evidence
            .precision;
        let updated_nodes = self.recompute_up(path, option, dimension);
        Ok(DiscountTrace {
            address: self.address_for_path(path),
            option: option.to_owned(),
            dimension: dimension.to_owned(),
            factor,
            precision_before,
            precision_after,
            updated_nodes,
        })
    }

    /// Collapse one retired non-root subtree into an exact Gaussian likelihood
    /// on its parent. This preserves ancestor and sibling posteriors for the
    /// current process-variance model while releasing descendant state.
    fn compact_subtree(
        &mut self,
        path: &[String],
        option: &str,
        dimension: &str,
    ) -> Result<CompactionTrace, FieldError> {
        if path.is_empty() {
            return Err(FieldError::RootCompaction);
        }
        let target_key = Self::key(path, option, dimension);
        let message = self
            .cells
            .get(&target_key)
            .map(|cell| cell.message_to_parent)
            .filter(|message| message.precision > 0.0)
            .ok_or(FieldError::NoEvidence)?;

        let compacted_keys: Vec<_> = self
            .cells
            .keys()
            .filter(|key| {
                key.option == option && key.dimension == dimension && key.path.starts_with(path)
            })
            .cloned()
            .collect();
        let observations = compacted_keys.iter().fold(0_u64, |total, key| {
            total
                .saturating_add(
                    self.cells
                        .get(key)
                        .map(|cell| cell.evidence.observations)
                        .unwrap_or(0),
                )
                .min(MAX_SAFE_OBSERVATIONS)
        });

        let parent_path = &path[..path.len() - 1];
        let parent_key = Self::key(parent_path, option, dimension);
        let parent = self
            .cells
            .get_mut(&parent_key)
            .expect("observed descendants have an ensured parent");
        if !(parent.evidence.precision + message.precision).is_finite()
            || !(parent.evidence.information + message.information).is_finite()
        {
            return Err(FieldError::InvalidObservation);
        }
        parent.child_messages = parent.child_messages.sub(message);
        parent.evidence.merge(Evidence {
            precision: message.precision,
            information: message.information,
            observations,
        });

        for key in &compacted_keys {
            self.cells.remove(key);
        }
        for (indexed_path, entries) in &mut self.path_index {
            if indexed_path.starts_with(path) {
                entries.remove(&(option.to_owned(), dimension.to_owned()));
            }
        }
        self.path_index.retain(|_, entries| !entries.is_empty());
        for ((indexed_path, indexed_option), dimensions) in &mut self.state_dimensions {
            if indexed_option == option && indexed_path.starts_with(path) {
                dimensions.remove(dimension);
            }
        }
        self.state_dimensions
            .retain(|_, dimensions| !dimensions.is_empty());
        self.index_coordinate(parent_path, option, dimension);
        self.process_variance_frozen = true;
        let updated_nodes = self.recompute_up(parent_path, option, dimension);

        Ok(CompactionTrace {
            address: self.address_for_path(path),
            compacted_into: self.address_for_path(parent_path),
            option: option.to_owned(),
            dimension: dimension.to_owned(),
            observations,
            removed_cells: compacted_keys.len(),
            updated_nodes,
        })
    }

    fn posterior_path(
        &self,
        path: &[String],
        option: &str,
        dimension: &str,
        work: &mut ResolveWork,
    ) -> Vec<ResidualTrace> {
        let root_key = Self::key(&[], option, dimension);
        work.parameter_lookups += 1;
        let root_cell = self.cells.get(&root_key);
        let root_prior = root_cell.and_then(|cell| cell.prior).unwrap_or(0.0);
        let mut external = Gaussian {
            mean: root_prior,
            variance: self.root_process_variance(dimension),
        };
        let mut posterior = combine(external, root_cell.map(Self::subtree).unwrap_or_default());
        let mut traces = vec![ResidualTrace {
            address: Address::new(),
            posterior_mean: posterior.mean,
            posterior_variance: posterior.variance,
            residual_mean: posterior.mean,
            declared_prior: root_prior,
            observations: root_cell
                .map(|cell| cell.evidence.observations)
                .unwrap_or(0),
        }];

        for depth in 0..path.len() {
            let parent_key = Self::key(&path[..depth], option, dimension);
            let child_key = Self::key(&path[..=depth], option, dimension);
            work.parameter_lookups += 2;
            let parent_cell = self.cells.get(&parent_key);
            let child_cell = self.cells.get(&child_key);
            let selected_message = child_cell
                .map(|cell| cell.message_to_parent)
                .unwrap_or_default();
            let outside_child = parent_cell
                .map(|cell| Self::subtree(cell).sub(selected_message))
                .unwrap_or_default();
            let parent_cavity = combine(external, outside_child);
            let child_prior = child_cell.and_then(|cell| cell.prior).unwrap_or(0.0);
            external = Gaussian {
                mean: parent_cavity.mean + child_prior,
                variance: parent_cavity.variance + self.process_variance(dimension, depth),
            };
            let previous_mean = posterior.mean;
            posterior = combine(external, child_cell.map(Self::subtree).unwrap_or_default());
            traces.push(ResidualTrace {
                address: self.address_for_path(&path[..=depth]),
                posterior_mean: posterior.mean,
                posterior_variance: posterior.variance,
                residual_mean: posterior.mean - previous_mean,
                declared_prior: child_prior,
                observations: child_cell
                    .map(|cell| cell.evidence.observations)
                    .unwrap_or(0),
            });
        }
        traces
    }

    fn candidates_on_path(
        &self,
        path: &[String],
        work: &mut ResolveWork,
    ) -> BTreeMap<String, BTreeSet<String>> {
        let mut candidates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for depth in 0..=path.len() {
            work.address_nodes += 1;
            let prefix = &path[..depth];
            let Some(entries) = self.path_index.get(prefix) else {
                continue;
            };
            work.stored_cells_visited += entries.len();
            for (option, dimension) in entries {
                if option != SHARED {
                    candidates
                        .entry(option.clone())
                        .or_default()
                        .insert(dimension.clone());
                }
            }
        }
        let options: Vec<String> = candidates.keys().cloned().collect();
        for depth in 0..=path.len() {
            let prefix = path[..depth].to_vec();
            for option in &options {
                work.parameter_lookups += 1;
                if let Some(dimensions) =
                    self.state_dimensions.get(&(prefix.clone(), option.clone()))
                {
                    work.stored_cells_visited += dimensions.len();
                    candidates
                        .entry(option.clone())
                        .or_default()
                        .extend(dimensions.iter().cloned());
                }
            }
        }
        candidates
    }

    /// Discover only explicitly allowed candidates without scanning unrelated
    /// options stored at shared ancestors. The allow-list restricts an
    /// addressed decision; it does not manufacture an absent candidate.
    fn candidates_among(
        &self,
        path: &[String],
        options: &BTreeSet<String>,
        work: &mut ResolveWork,
    ) -> BTreeMap<String, BTreeSet<String>> {
        let mut candidates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for depth in 0..=path.len() {
            work.address_nodes += 1;
            let prefix = path[..depth].to_vec();
            let Some(bucket) = self.path_index.get(&prefix) else {
                continue;
            };
            for option in options {
                work.parameter_lookups += 1;
                let Some(dimensions) = self.state_dimensions.get(&(prefix.clone(), option.clone()))
                else {
                    continue;
                };
                for dimension in dimensions {
                    work.parameter_lookups += 1;
                    if !bucket.contains(&(option.clone(), dimension.clone())) {
                        continue;
                    }
                    work.stored_cells_visited += 1;
                    candidates
                        .entry(option.clone())
                        .or_default()
                        .insert(dimension.clone());
                }
            }
        }
        for depth in 0..=path.len() {
            let prefix = path[..depth].to_vec();
            for (option, target) in &mut candidates {
                work.parameter_lookups += 1;
                let Some(dimensions) = self.state_dimensions.get(&(prefix.clone(), option.clone()))
                else {
                    continue;
                };
                work.stored_cells_visited += dimensions.len();
                target.extend(dimensions.iter().cloned());
            }
        }
        candidates
    }

    fn weight_on_path(
        &self,
        path: &[String],
        option: &str,
        dimension: &str,
        work: &mut ResolveWork,
    ) -> f64 {
        let mut sum = 0.0;
        let mut found = false;
        for depth in 0..=path.len() {
            for candidate in [SHARED, option] {
                work.parameter_lookups += 1;
                let key = Self::key(&path[..depth], candidate, dimension);
                if let Some(weight) = self.cells.get(&key).and_then(|cell| cell.weight) {
                    sum += weight;
                    found = true;
                }
            }
        }
        if found {
            sum
        } else {
            1.0
        }
    }

    fn gate_on_path(
        &self,
        path: &[String],
        option: &str,
        dimension: &str,
        work: &mut ResolveWork,
    ) -> bool {
        for depth in 0..=path.len() {
            work.parameter_lookups += 1;
            let key = Self::key(&path[..depth], option, dimension);
            if self.cells.get(&key).and_then(|cell| cell.gate) == Some(false) {
                return false;
            }
        }
        true
    }

    fn numeric_on_path(
        &self,
        path: &[String],
        option: &str,
        dimension: &str,
        work: &mut ResolveWork,
    ) -> bool {
        for depth in 0..=path.len() {
            work.parameter_lookups += 1;
            let key = Self::key(&path[..depth], option, dimension);
            if self.cells.get(&key).is_some_and(|cell| {
                cell.prior.is_some()
                    || cell.evidence.precision > 0.0
                    || cell.evidence.observations > 0
                    || cell.child_messages.precision > 0.0
            }) {
                return true;
            }
        }
        false
    }

    fn decide_candidates(
        &self,
        path: &[String],
        policy: DecisionPolicy,
        candidates: BTreeMap<String, BTreeSet<String>>,
        mut work: ResolveWork,
    ) -> FieldDecision {
        let temperature = if policy.temperature.is_finite() {
            policy.temperature.max(0.0)
        } else {
            0.0
        };
        let policy = DecisionPolicy { temperature };
        let mut alternatives = Vec::new();
        for (option, dimensions) in candidates {
            let mut viable = true;
            let mut expected_score = 0.0;
            let mut score_variance = 0.0;
            let mut traces = Vec::new();
            for dimension in dimensions {
                if !self.gate_on_path(path, &option, &dimension, &mut work) {
                    viable = false;
                }
                let numeric = self.numeric_on_path(path, &option, &dimension, &mut work);
                let contributions = if numeric {
                    self.posterior_path(path, &option, &dimension, &mut work)
                } else {
                    Vec::new()
                };
                let (mean, variance) = contributions
                    .last()
                    .map(|trace| (trace.posterior_mean, trace.posterior_variance))
                    .unwrap_or((0.0, 0.0));
                let weight = if numeric {
                    self.weight_on_path(path, &option, &dimension, &mut work)
                } else {
                    0.0
                };
                let contribution = mean * weight;
                expected_score += contribution;
                score_variance += variance * weight * weight;
                traces.push(FieldDimensionTrace {
                    dimension,
                    mean,
                    variance,
                    weight,
                    contribution,
                    contributions,
                });
            }
            let decision_score = expected_score - temperature * score_variance.sqrt();
            alternatives.push(FieldResolvedOption {
                option,
                viable,
                expected_score,
                score_variance,
                decision_score,
                dimensions: traces,
            });
        }
        alternatives.sort_by(|left, right| {
            right
                .viable
                .cmp(&left.viable)
                .then_with(|| fcmp(left.decision_score, right.decision_score))
                .then_with(|| left.option.cmp(&right.option))
        });
        work.ranked_options = alternatives.len();
        FieldDecision {
            address: self.address_for_path(path),
            policy,
            alternatives,
            work,
        }
    }

    pub fn decide(
        &self,
        input: &Address,
        policy: DecisionPolicy,
    ) -> Result<FieldDecision, FieldError> {
        let path = self.path(input)?;
        let mut work = ResolveWork::default();
        let candidates = self.candidates_on_path(&path, &mut work);
        Ok(self.decide_candidates(&path, policy, candidates, work))
    }

    /// Resolve only options allowed by the current host/product configuration.
    pub fn decide_among<S: AsRef<str>>(
        &self,
        input: &Address,
        options: impl IntoIterator<Item = S>,
        policy: DecisionPolicy,
    ) -> Result<FieldDecision, FieldError> {
        let path = self.path(input)?;
        let options = options
            .into_iter()
            .map(|option| option.as_ref().to_owned())
            .collect::<BTreeSet<_>>();
        let mut work = ResolveWork::default();
        let candidates = self.candidates_among(&path, &options, &mut work);
        Ok(self.decide_candidates(&path, policy, candidates, work))
    }

    pub fn resolve(
        &self,
        input: &Address,
        policy: DecisionPolicy,
    ) -> Result<Vec<String>, FieldError> {
        Ok(self
            .decide(input, policy)?
            .alternatives
            .into_iter()
            .filter(|candidate| candidate.viable)
            .map(|candidate| candidate.option)
            .collect())
    }

    pub fn receipt(
        decision: FieldDecision,
        tensor_revision: impl Into<String>,
        selected: impl IntoIterator<Item = impl Into<String>>,
    ) -> RoutingReceipt {
        RoutingReceipt {
            tensor_revision: tensor_revision.into(),
            decision,
            selected: selected.into_iter().map(Into::into).collect(),
        }
    }

    pub fn cells(&self) -> Vec<JsonFieldCell> {
        self.cells
            .iter()
            .filter(|(_, cell)| {
                cell.prior.is_some()
                    || cell.weight.is_some()
                    || cell.gate.is_some()
                    || !is_empty_evidence(&cell.evidence)
            })
            .map(|(key, cell)| JsonFieldCell {
                address: self.address_for_path(&key.path),
                option: key.option.clone(),
                dimension: key.dimension.clone(),
                prior: cell.prior,
                weight: cell.weight,
                gate: cell.gate,
                evidence: cell.evidence,
            })
            .collect()
    }

    /// Number of materialized internal parameter nodes, including zero-state
    /// prefixes required by cached message passing.
    pub fn stored_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Exact compressed Gaussian message this addressed subtree sends upward.
    pub fn message_to_parent(
        &self,
        input: &Address,
        option: &str,
        dimension: &str,
    ) -> Result<Natural, FieldError> {
        let path = self.path(input)?;
        if path.is_empty() {
            return Err(FieldError::AddressTooShallow);
        }
        Ok(self
            .cells
            .get(&Self::key(&path, option, dimension))
            .map(|cell| cell.message_to_parent)
            .unwrap_or_default())
    }

    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            layers: &'a [String],
            default_root_process_variance: f64,
            default_process_variance: &'a [f64],
            dimension_root_process_variance: &'a BTreeMap<String, f64>,
            dimension_process_variance: &'a BTreeMap<String, Vec<f64>>,
            process_variance_frozen: bool,
            cells: Vec<JsonFieldCell>,
        }
        serde_json::to_string(&Snapshot {
            layers: &self.layers,
            default_root_process_variance: self.default_root_process_variance,
            default_process_variance: &self.default_process_variance,
            dimension_root_process_variance: &self.dimension_root_process_variance,
            dimension_process_variance: &self.dimension_process_variance,
            process_variance_frozen: self.process_variance_frozen,
            cells: self.cells(),
        })
        .expect("field snapshots contain only serializable state")
    }

    pub fn from_json(snapshot: &str) -> Option<Self> {
        #[derive(Deserialize)]
        struct Snapshot {
            layers: Vec<String>,
            #[serde(default = "default_root_process_variance")]
            default_root_process_variance: f64,
            default_process_variance: Vec<f64>,
            #[serde(default)]
            dimension_root_process_variance: BTreeMap<String, f64>,
            #[serde(default)]
            dimension_process_variance: BTreeMap<String, Vec<f64>>,
            #[serde(default)]
            process_variance_frozen: bool,
            cells: Vec<JsonFieldCell>,
        }
        let snapshot: Snapshot = serde_json::from_str(snapshot).ok()?;
        let unique_layers: BTreeSet<_> = snapshot.layers.iter().collect();
        if snapshot.layers.iter().any(|layer| layer.is_empty())
            || unique_layers.len() != snapshot.layers.len()
            || !snapshot.default_root_process_variance.is_finite()
            || snapshot.default_root_process_variance <= 0.0
            || snapshot
                .dimension_root_process_variance
                .values()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || snapshot.default_process_variance.len() != snapshot.layers.len()
            || snapshot
                .default_process_variance
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || snapshot.dimension_process_variance.values().any(|values| {
                values.len() != snapshot.layers.len()
                    || values
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
            })
        {
            return None;
        }
        let mut field = Self::new(snapshot.layers);
        field.default_root_process_variance = snapshot.default_root_process_variance;
        field.default_process_variance = snapshot.default_process_variance;
        field.dimension_root_process_variance = snapshot.dimension_root_process_variance;
        field.dimension_process_variance = snapshot.dimension_process_variance;
        field.process_variance_frozen = snapshot.process_variance_frozen;
        let mut seen = BTreeSet::new();
        for cell in snapshot.cells {
            let path = field.path(&cell.address).ok()?;
            if cell.prior.is_some_and(|value| !value.is_finite())
                || cell.weight.is_some_and(|value| !value.is_finite())
                || !cell.evidence.precision.is_finite()
                || cell.evidence.precision < 0.0
                || !cell.evidence.information.is_finite()
                || (cell.evidence.precision == 0.0 && cell.evidence.information != 0.0)
                || cell.evidence.observations > MAX_SAFE_OBSERVATIONS
            {
                return None;
            }
            let key = Self::key(&path, &cell.option, &cell.dimension);
            if !seen.insert(key.clone()) {
                return None;
            }
            field.ensure_path(&path, &cell.option, &cell.dimension);
            field.index_coordinate(&path, &cell.option, &cell.dimension);
            let target = field.cells.get_mut(&key)?;
            target.prior = cell.prior;
            target.weight = cell.weight;
            target.gate = cell.gate;
            target.evidence = cell.evidence;
        }
        field.rebuild_messages();
        Some(field)
    }
}

/// A structurally attenuated write capability. Feedback may update derived
/// ancestor messages, but the supplied evidence remains attached at or below
/// the writer's authority floor.
pub struct FieldWriter<'a> {
    field: &'a mut AddressedField,
    floor: Option<usize>,
}

impl FieldWriter<'_> {
    fn checked_path(&self, input: &Address) -> Result<Vec<String>, FieldError> {
        let path = self.field.path(input)?;
        if let Some(floor) = self.floor {
            if path.is_empty() || path.len() - 1 < floor {
                return Err(FieldError::WriteUp);
            }
        }
        Ok(path)
    }

    pub fn set_prior(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        value: f64,
    ) -> Result<(), FieldError> {
        let path = self.checked_path(input)?;
        self.field.set_prior(&path, option, dimension, value)
    }

    pub fn set_weight(
        &mut self,
        input: &Address,
        dimension: &str,
        value: f64,
    ) -> Result<(), FieldError> {
        let path = self.checked_path(input)?;
        self.field.set_weight(&path, SHARED, dimension, value)
    }

    pub fn set_option_weight(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        value: f64,
    ) -> Result<(), FieldError> {
        let path = self.checked_path(input)?;
        self.field.set_weight(&path, option, dimension, value)
    }

    pub fn set_gate(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        viable: bool,
    ) -> Result<(), FieldError> {
        let path = self.checked_path(input)?;
        self.field.set_gate(&path, option, dimension, viable)
    }

    pub fn observe(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        value: f64,
        variance: f64,
    ) -> Result<FeedbackTrace, FieldError> {
        let path = self.checked_path(input)?;
        let before = self
            .field
            .posterior_path(&path, option, dimension, &mut ResolveWork::default())
            .last()
            .map(|trace| (trace.posterior_mean, trace.posterior_variance));
        let evidence = Evidence::from_observation(value, variance)?;
        let updated_nodes = self
            .field
            .merge_evidence(&path, option, dimension, evidence)?;
        let after = self
            .field
            .posterior_path(&path, option, dimension, &mut ResolveWork::default())
            .last()
            .map(|trace| (trace.posterior_mean, trace.posterior_variance))
            .expect("observed path has a posterior");
        Ok(FeedbackTrace {
            address: self.field.address_for_path(&path),
            option: option.to_owned(),
            dimension: dimension.to_owned(),
            observed: value,
            observation_variance: variance,
            updated_nodes,
            posterior_before: before,
            posterior_after: after,
        })
    }

    pub fn merge_evidence(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        evidence: Evidence,
    ) -> Result<usize, FieldError> {
        let path = self.checked_path(input)?;
        self.field
            .merge_evidence(&path, option, dimension, evidence)
    }

    /// Reduce the effective precision of evidence at one exact address. The
    /// product supplies the clock-derived factor; Sporewright remains clock-free.
    pub fn discount_evidence(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
        factor: f64,
    ) -> Result<DiscountTrace, FieldError> {
        let path = self.checked_path(input)?;
        self.field
            .discount_evidence(&path, option, dimension, factor)
    }

    /// Compact a retired subtree into its parent's sufficient statistics. The
    /// writer must have authority over the parent because this is an explicit
    /// integrity upgrade, not ordinary participant feedback.
    pub fn compact_subtree(
        &mut self,
        input: &Address,
        option: &str,
        dimension: &str,
    ) -> Result<CompactionTrace, FieldError> {
        let path = self.field.path(input)?;
        if path.is_empty() {
            return Err(FieldError::RootCompaction);
        }
        if let Some(floor) = self.floor {
            if path.len() < 2 || path.len() - 2 < floor {
                return Err(FieldError::WriteUp);
            }
        }
        self.field.compact_subtree(&path, option, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_schemas_and_snapshots_reject_ambiguous_state() {
        assert_eq!(
            AddressedField::try_new(["root", "root"]).unwrap_err(),
            FieldError::InvalidSchema
        );
        assert!(AddressedField::from_json(
            r#"{"layers":["root"],"default_process_variance":[1.0],"cells":[{"address":{"root":"x"},"option":"o","dimension":"q","evidence":{"precision":0.0,"information":1.0,"observations":1}}]}"#
        )
        .is_none());
    }

    #[test]
    fn implicit_root_is_a_complete_field_and_layer_writers_cannot_write_up() {
        let mut field = AddressedField::new(Vec::<String>::new());
        field
            .root_writer()
            .set_prior(&Address::new(), "local", "cost", 2.0)
            .unwrap();
        let decision = field
            .decide(&Address::new(), DecisionPolicy::default())
            .unwrap();
        assert_eq!(decision.alternatives[0].option, "local");
        assert_eq!(
            decision.alternatives[0].dimensions[0].contributions[0].address,
            Address::new()
        );

        let mut layered = AddressedField::new(["workspace"]);
        assert_eq!(
            layered
                .writer("workspace")
                .unwrap()
                .set_prior(&Address::new(), "local", "cost", 0.0),
            Err(FieldError::WriteUp)
        );
    }

    #[test]
    fn path_residuals_add_instead_of_overriding() {
        let mut field = AddressedField::new(["b", "c"]);
        let root = Address::new();
        let b = address([("b", "B1")]);
        let c = address([("b", "B1"), ("c", "C11")]);
        let mut writer = field.root_writer();
        writer.set_prior(&root, "route", "q1", 1.0).unwrap();
        writer.set_prior(&b, "route", "q1", 2.0).unwrap();
        writer.set_prior(&c, "route", "q1", 3.0).unwrap();

        let decision = field.decide(&c, DecisionPolicy::default()).unwrap();
        let q1 = &decision.alternatives[0].dimensions[0];
        assert!((q1.mean - 6.0).abs() < 1e-12);
        assert_eq!(
            q1.contributions
                .iter()
                .map(|term| term.declared_prior)
                .collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn feedback_updates_one_path_and_shared_ancestors() {
        let mut field = AddressedField::new(["source", "environment"]);
        field.set_default_root_process_variance(1.0).unwrap();
        for layer in ["source", "environment"] {
            field.set_default_process_variance(layer, 1.0).unwrap();
        }
        let root = Address::new();
        let a = address([("source", "a"), ("environment", "browser")]);
        let b = address([("source", "b"), ("environment", "browser")]);
        field
            .root_writer()
            .set_prior(&root, "browser", "failure", 0.0)
            .unwrap();
        field
            .writer("environment")
            .unwrap()
            .observe(&a, "browser", "failure", 1.0, 0.1)
            .unwrap();

        let a_mean = field
            .decide(&a, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .mean;
        let b_mean = field
            .decide(&b, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .mean;
        assert!(
            a_mean > b_mean,
            "the observed branch must specialize its sibling"
        );
        assert!(
            b_mean > 0.0,
            "the sibling must inherit bottom-up root learning"
        );
    }

    #[test]
    fn unrelated_population_does_not_change_resolution_work() {
        let mut field = AddressedField::new(["workspace", "task"]);
        let root = Address::new();
        field
            .root_writer()
            .set_prior(&root, "fast", "cost", 0.0)
            .unwrap();
        let target = address([("workspace", "target"), ("task", "one")]);
        let baseline = field
            .decide(&target, DecisionPolicy::default())
            .unwrap()
            .work;

        for index in 0..10_000 {
            let other = address([("workspace", &format!("w{index}")), ("task", "one")]);
            field
                .writer("task")
                .unwrap()
                .set_prior(&other, "slow", "cost", 1.0)
                .unwrap();
        }
        let populated = field
            .decide(&target, DecisionPolicy::default())
            .unwrap()
            .work;
        assert_eq!(baseline, populated);
    }

    #[test]
    fn decide_among_bounds_candidates_and_work_to_the_host_allow_list() {
        let mut field = AddressedField::new(["workspace", "task"]);
        let root = Address::new();
        let target = address([("workspace", "target"), ("task", "one")]);
        field
            .root_writer()
            .set_prior(&root, "allowed", "cost", 0.0)
            .unwrap();
        for index in 0..1_000 {
            field
                .root_writer()
                .set_prior(
                    &root,
                    &format!("historical-{index}"),
                    "cost",
                    index as f64 + 1.0,
                )
                .unwrap();
        }

        let decision = field
            .decide_among(&target, ["allowed", "absent"], DecisionPolicy::default())
            .unwrap();
        assert_eq!(
            decision
                .alternatives
                .iter()
                .map(|candidate| candidate.option.as_str())
                .collect::<Vec<_>>(),
            vec!["allowed"]
        );
        assert_eq!(decision.work.ranked_options, 1);
        assert!(decision.work.stored_cells_visited < 20);
    }

    #[test]
    fn hard_veto_cannot_be_cancelled_by_curiosity() {
        let mut field = AddressedField::new(["environment"]);
        let root = Address::new();
        let local = address([("environment", "browser")]);
        let mut writer = field.root_writer();
        writer.set_prior(&root, "browser", "cost", 0.0).unwrap();
        writer.set_prior(&root, "container", "cost", 1.0).unwrap();
        writer
            .set_gate(&local, "browser", "reachable", false)
            .unwrap();
        let resolved = field
            .resolve(
                &local,
                DecisionPolicy {
                    temperature: 1_000.0,
                },
            )
            .unwrap();
        assert_eq!(resolved, vec!["container"]);
    }

    #[test]
    fn snapshot_rebuilds_derived_messages() {
        let mut field = AddressedField::new(["leaf"]);
        let root = Address::new();
        let leaf = address([("leaf", "a")]);
        field
            .root_writer()
            .set_prior(&root, "route", "cost", 0.0)
            .unwrap();
        field
            .writer("leaf")
            .unwrap()
            .observe(&leaf, "route", "cost", 2.0, 0.5)
            .unwrap();
        let rebuilt = AddressedField::from_json(&field.to_json()).unwrap();
        assert_eq!(field, rebuilt);
        assert_eq!(field.to_json(), rebuilt.to_json());
    }

    #[test]
    fn retired_subtree_compacts_exactly_for_ancestors_and_siblings() {
        let mut field = AddressedField::new(["source", "task", "environment"]);
        let root = Address::new();
        let retired = address([("source", "portal"), ("task", "old")]);
        let observed = address([
            ("source", "portal"),
            ("task", "old"),
            ("environment", "browser"),
        ]);
        let sibling = address([
            ("source", "portal"),
            ("task", "new"),
            ("environment", "browser"),
        ]);
        {
            let mut writer = field.root_writer();
            writer.set_prior(&root, "browser", "failure", 0.0).unwrap();
            writer
                .observe(&observed, "browser", "failure", 8.0, 0.2)
                .unwrap();
        }
        let before = field
            .decide(&sibling, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .clone();
        let cells_before = field.stored_cell_count();

        let trace = field
            .root_writer()
            .compact_subtree(&retired, "browser", "failure")
            .unwrap();
        let after = field
            .decide(&sibling, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .clone();

        assert!((before.mean - after.mean).abs() < 1e-12);
        assert!((before.variance - after.variance).abs() < 1e-12);
        assert_eq!(trace.observations, 1);
        assert_eq!(trace.removed_cells, 2);
        assert!(field.stored_cell_count() < cells_before);
        assert_eq!(
            field.set_default_process_variance("task", 2.0),
            Err(FieldError::FrozenProcessVariance)
        );
        let mut rebuilt = AddressedField::from_json(&field.to_json()).unwrap();
        assert_eq!(
            rebuilt.set_default_process_variance("task", 2.0),
            Err(FieldError::FrozenProcessVariance)
        );
        let rebuilt_dimension = rebuilt
            .decide(&sibling, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .clone();
        assert!((after.mean - rebuilt_dimension.mean).abs() < 1e-12);
        assert!((after.variance - rebuilt_dimension.variance).abs() < 1e-12);
    }

    #[test]
    fn discount_reopens_uncertainty_without_a_clock_in_the_core() {
        let mut field = AddressedField::new(["leaf"]);
        let root = Address::new();
        let leaf = address([("leaf", "changing")]);
        {
            let mut writer = field.root_writer();
            writer.set_prior(&root, "route", "cost", 0.0).unwrap();
            writer.observe(&leaf, "route", "cost", 4.0, 0.01).unwrap();
        }
        let before = field
            .decide(&leaf, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .clone();
        let trace = field
            .root_writer()
            .discount_evidence(&leaf, "route", "cost", 0.0)
            .unwrap();
        let after = field
            .decide(&leaf, DecisionPolicy::default())
            .unwrap()
            .alternatives[0]
            .dimensions[0]
            .clone();
        assert!(after.variance > before.variance);
        assert!(after.mean.abs() < before.mean.abs());
        assert_eq!(trace.precision_after, 0.0);
        assert_eq!(
            field
                .root_writer()
                .discount_evidence(&leaf, "route", "cost", 1.1),
            Err(FieldError::InvalidDiscount)
        );
    }

    fn dense_inverse(mut matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let size = matrix.len();
        let mut inverse = vec![vec![0.0; size]; size];
        for (index, row) in inverse.iter_mut().enumerate() {
            row[index] = 1.0;
        }
        for column in 0..size {
            let pivot = (column..size)
                .max_by(|left, right| {
                    matrix[*left][column]
                        .abs()
                        .partial_cmp(&matrix[*right][column].abs())
                        .unwrap()
                })
                .unwrap();
            matrix.swap(column, pivot);
            inverse.swap(column, pivot);
            let scale = matrix[column][column];
            for entry in &mut matrix[column] {
                *entry /= scale;
            }
            for entry in &mut inverse[column] {
                *entry /= scale;
            }
            for row in 0..size {
                if row == column {
                    continue;
                }
                let factor = matrix[row][column];
                for inner in 0..size {
                    matrix[row][inner] -= factor * matrix[column][inner];
                    inverse[row][inner] -= factor * inverse[column][inner];
                }
            }
        }
        inverse
    }

    #[test]
    fn sparse_messages_match_a_dense_branched_gaussian_posterior() {
        // Nodes: A=0, B1=1, B2=2, C11=3, C21=4.
        let parents = [None, Some(0), Some(0), Some(1), Some(2)];
        let residuals: [f64; 5] = [1.0, 2.0, 2.0, 3.0, 3.0];
        let variances: [f64; 5] = [0.7, 1.1, 1.1, 0.5, 0.5];
        let observations: [(usize, f64, f64); 3] = [(1, 5.0, 1.0), (3, 10.0, 0.2), (4, -1.0, 0.4)];
        let size = parents.len();
        let mut precision = vec![vec![0.0; size]; size];
        let mut information = vec![0.0; size];
        for index in 0..size {
            let inverse_variance = variances[index].recip();
            if let Some(parent) = parents[index] {
                precision[index][index] += inverse_variance;
                precision[parent][parent] += inverse_variance;
                precision[index][parent] -= inverse_variance;
                precision[parent][index] -= inverse_variance;
                information[index] += residuals[index] * inverse_variance;
                information[parent] -= residuals[index] * inverse_variance;
            } else {
                precision[index][index] += inverse_variance;
                information[index] += residuals[index] * inverse_variance;
            }
        }
        for (node, observed, variance) in observations {
            let observation_precision = 1.0 / variance;
            precision[node][node] += observation_precision;
            information[node] += observed * observation_precision;
        }
        let covariance = dense_inverse(precision);
        let dense_mean: Vec<f64> = covariance
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&information)
                    .map(|(value, info)| value * info)
                    .sum()
            })
            .collect();

        let mut field = AddressedField::new(["b", "c"]);
        field
            .set_dimension_root_process_variance("q", variances[0])
            .unwrap();
        field
            .set_dimension_process_variance("q", "b", variances[1])
            .unwrap();
        field
            .set_dimension_process_variance("q", "c", variances[3])
            .unwrap();
        let root = Address::new();
        let b1 = address([("b", "B1")]);
        let b2 = address([("b", "B2")]);
        let c11 = address([("b", "B1"), ("c", "C11")]);
        let c21 = address([("b", "B2"), ("c", "C21")]);
        {
            let mut writer = field.root_writer();
            writer.set_prior(&root, "route", "q", residuals[0]).unwrap();
            writer.set_prior(&b1, "route", "q", residuals[1]).unwrap();
            writer.set_prior(&b2, "route", "q", residuals[2]).unwrap();
            writer.set_prior(&c11, "route", "q", residuals[3]).unwrap();
            writer.set_prior(&c21, "route", "q", residuals[4]).unwrap();
            writer.observe(&b1, "route", "q", 5.0, 1.0).unwrap();
            writer.observe(&c11, "route", "q", 10.0, 0.2).unwrap();
            writer.observe(&c21, "route", "q", -1.0, 0.4).unwrap();
        }
        for (query, node) in [(&c11, 3usize), (&c21, 4usize)] {
            let decision = field.decide(query, DecisionPolicy::default()).unwrap();
            let posterior = &decision.alternatives[0].dimensions[0];
            assert!(
                (posterior.mean - dense_mean[node]).abs() < 1e-10,
                "node {node}: sparse mean {} != dense mean {}",
                posterior.mean,
                dense_mean[node]
            );
            assert!(
                (posterior.variance - covariance[node][node]).abs() < 1e-10,
                "node {node}: sparse variance {} != dense variance {}",
                posterior.variance,
                covariance[node][node]
            );
        }
    }
}
