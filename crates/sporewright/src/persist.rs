// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//! Persistence — the durability/versioning port.
//!
//! The tensor core is clock-free; **the database is the clock.** This port is where
//! a product's facts live and where versioning lives — an edge KV or a relational DB
//! for a server product, SQLite/a file natively, IndexedDB in a browser, or memory.
//! Glue, not core.
//!
//! A real impl stores [`Tensor::cells`](crate::Tensor::cells) as versioned rows and
//! answers "everything since version N" (a `WHERE version > N` query, CDN-cacheable
//! via an immutable-clock URL). The client overlays the returned cells with
//! [`Tensor::apply_json`](crate::Tensor::apply_json). Offline writes buffer locally
//! and flush on reconnect; disjoint coordinates + the DB's upsert/CAS resolve the
//! rare same-row conflict. [`MemStore`] is the trivial reference: a whole-snapshot
//! string, no versioning — enough for tests and ephemeral meshes.

/// The persistence port: load and save a serialized tensor (`Tensor::to_json`).
pub trait Store {
    /// The last saved snapshot, if any.
    fn load(&self) -> Option<String>;
    /// Persist a snapshot (typically [`Tensor::to_json`](crate::Tensor::to_json)).
    fn save(&mut self, snapshot: &str);
}

/// An in-memory [`Store`] — the reference implementation. Real durability +
/// versioning is the product's database.
#[derive(Clone, Debug, Default)]
pub struct MemStore {
    snap: Option<String>,
}

impl Store for MemStore {
    fn load(&self) -> Option<String> {
        self.snap.clone()
    }
    fn save(&mut self, snapshot: &str) {
        self.snap = Some(snapshot.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Cursor, Tensor};

    #[test]
    fn a_snapshot_survives_a_store() {
        let mut a = Tensor::new(["global", "device"]);
        a.writer("device")
            .unwrap()
            .set_value("device", "A", "groq", "latency", 0.1)
            .unwrap();
        let mut o = a.writer("global").unwrap();
        o.set_weight("global", "", "latency", 1.0).unwrap();
        o.set_bytes("global", "", "doc:cv", "state", vec![1, 2, 3])
            .unwrap();

        let mut store = MemStore::default();
        store.save(&a.to_json());
        let b = Tensor::from_json(&store.load().unwrap()).unwrap();

        let at: Cursor = [("device".to_string(), "A".to_string())]
            .into_iter()
            .collect();
        assert_eq!(b.value(&at, "groq", "latency"), Some(0.1));
        assert_eq!(b.weight(&Cursor::new(), "groq", "latency"), Some(1.0));
        assert_eq!(
            b.bytes(&Cursor::new(), "doc:cv", "state"),
            Some([1u8, 2, 3].as_slice())
        );
    }

    #[test]
    fn from_json_rejects_garbage() {
        assert!(Tensor::from_json("not json").is_none());
        assert!(Tensor::from_json("{}").is_none()); // no levels
    }
}
