// Writing our own implementation with logic from else crate
// https://github.com/Manishearth/elsa/blob/master/src/sync.rs

use hashbrown::{Equivalent, HashMap};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

pub(crate) struct AppendHashMap<K, V> {
    map: RwLock<HashMap<K, Arc<V>>>,
}

impl<K: Eq + Hash, V> AppendHashMap<K, V> {
    pub fn get<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Equivalent<K>,
    {
        let map = self.map.read().unwrap();
        map.get(key).map(|v| v.clone())
    }

    pub fn insert(&self, k: K, v: V) -> Arc<V> {
        let mut map = self.map.write().unwrap();
        let value = Arc::new(v);
        map.entry(k).or_insert(value.clone());
        value
    }

    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }
}
