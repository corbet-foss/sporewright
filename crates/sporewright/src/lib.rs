// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! # sporewright
//!
//! Sparse addressed residual fields for resilient pull-based work distribution.
//! Every prefix of a product-declared address adds policy and belief to a route.
//! Cached Gaussian messages carry observations from one realized leaf back toward
//! shared ancestors in `O(address depth)` work.
//!
//! See `docs/MODEL.md` for the algebra and product boundary.
//!
//! ```
//! use sporewright::field::{address, AddressedField, DecisionPolicy};
//!
//! let mut field = AddressedField::new(["system", "environment"]);
//! let root = address([("system", "careervector")]);
//! let browser = address([("system", "careervector"), ("environment", "browser")]);
//! let mut writer = field.writer("system").unwrap();
//! writer.set_prior(&root, "fast", "cost", 1.0).unwrap();
//! writer.set_prior(&root, "careful", "cost", 2.0).unwrap();
//! writer.observe(&browser, "fast", "cost", 4.0, 0.1).unwrap();
//! let decision = field.decide(&browser, DecisionPolicy::default()).unwrap();
//! assert_eq!(decision.alternatives[0].option, "careful");
//! ```

pub mod budget;
pub mod config;
pub mod explore;
pub mod field;
pub mod normalize;
pub mod persist;
pub mod schedule;
pub mod sync;
pub mod tensor;
pub mod trust;

pub use budget::{budget_dim, Budget, Pool, TokenBucket, BUDGET_PREFIX, DEFAULT_STEP, MIN_STEP};
pub use config::{instantiate, SeedCell, SeedKind, TensorConfig};
pub use explore::{
    plan_exploration, BatchExplorer, ExecutionPlan, ExplorationBudget, PlanError, PlannedExecution,
};
pub use normalize::{max_abs_scale, Norm};
pub use persist::{MemStore, Store};
pub use schedule::{
    backlog_saturation, budget_from_saturation, lease_in_flight, priority, priority_bucket,
    DEFAULT_LEASE_TTL_SECS,
};
pub use sync::{gate_capabilities, gate_dim, Device, GATE_PREFIX, REACH};
pub use tensor::{
    scope, Cursor, DimensionTrace, JsonCell, Origin, PreferenceChange, ResolvedOption, Scope,
    Tensor, Value, WriteError, Writer, SHARED,
};
pub use trust::{
    clamp01, next_churn, next_trust_state, reinvestigation_priority, should_reinvestigate,
    time_weight, update_pair_trust, ChurnState, ChurnTimestamp, PairTrustResult, PreviousChurn,
    TimestampSource,
};
