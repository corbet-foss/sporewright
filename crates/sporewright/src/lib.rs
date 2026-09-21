// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! # sporewright
//!
//! One object — a sparse tensor of facts at `(level, instance, option, dimension)`,
//! each cell two co-located layers (a `value` and a `weight`) — and the operations
//! on it. A product declares its axes and feeds facts; the engine **folds** the
//! context tiers (most-specific wins), **resolves** the dimension axis into the
//! ordered queue (`Σ value·weight`), **reduces** measured values up, and
//! **corroborates** attested options up. The core is **clock-free**: versioning and
//! durability are the persistence port (the product's database).
//!
//! See `docs/MODEL.md` for the model, the maths, and the design choices.
//!
//! ```
//! use sporewright::tensor::{Tensor, Cursor};
//!
//! let mut t = Tensor::new(["global", "device"]);
//! let mut w = t.writer("global").unwrap();
//! w.set_value("global", "", "groq", "latency", 0.4).unwrap();
//! w.set_value("global", "", "cerebras", "latency", 0.3).unwrap();
//! w.set_weight("global", "", "latency", 1.0).unwrap();
//! assert_eq!(t.resolve(&Cursor::new()), vec!["cerebras".to_string(), "groq".to_string()]);
//! ```

pub mod budget;
pub mod config;
pub mod normalize;
pub mod persist;
pub mod schedule;
pub mod sync;
pub mod tensor;

pub use budget::{budget_dim, Budget, Pool, TokenBucket, BUDGET_PREFIX, DEFAULT_STEP, MIN_STEP};
pub use config::{instantiate, SeedCell, SeedKind, TensorConfig};
pub use normalize::{max_abs_scale, Norm};
pub use persist::{MemStore, Store};
pub use schedule::{
    backlog_saturation, budget_from_saturation, lease_in_flight, priority, priority_bucket,
    DEFAULT_LEASE_TTL_SECS,
};
pub use sync::{gate_capabilities, gate_dim, Device, GATE_PREFIX, REACH};
pub use tensor::{Cursor, JsonCell, Tensor, Value, WriteError, Writer, SHARED};
