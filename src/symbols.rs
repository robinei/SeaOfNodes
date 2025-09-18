use std::sync::{RwLock, OnceLock};
use std::collections::HashMap;
use std::num::NonZeroU32;

// Symbol interning for efficient string handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(NonZeroU32);

struct SymbolTable {
    strings: Vec<String>,
    indices: HashMap<String, SymbolId>,
}

impl SymbolTable {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            indices: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> SymbolId {
        if let Some(&id) = self.indices.get(s) {
            id
        } else {
            // Use 1-based indexing to ensure NonZeroU32 is always valid
            let index = self.strings.len() as u32 + 1;
            let id = SymbolId(NonZeroU32::new(index).unwrap());
            self.strings.push(s.to_string());
            self.indices.insert(s.to_string(), id);
            id
        }
    }

    fn get(&self, id: SymbolId) -> Option<&str> {
        let index = id.0.get() as usize - 1; // Convert back to 0-based
        self.strings.get(index).map(|s| s.as_str())
    }
}

static SYMBOL_TABLE: OnceLock<RwLock<SymbolTable>> = OnceLock::new();

pub fn intern_symbol(s: &str) -> SymbolId {
    let table = SYMBOL_TABLE.get_or_init(|| RwLock::new(SymbolTable::new()));
    
    // Try read lock first for existing symbols
    {
        let read_guard = table.read().unwrap();
        if let Some(&id) = read_guard.indices.get(s) {
            return id;
        }
    }
    
    // Need write lock to add new symbol
    let mut write_guard = table.write().unwrap();
    write_guard.intern(s)
}

pub fn symbol_name(id: SymbolId) -> Option<String> {
    let table = SYMBOL_TABLE.get_or_init(|| RwLock::new(SymbolTable::new()));
    let read_guard = table.read().unwrap();
    read_guard.get(id).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_interning() {
        let a1 = intern_symbol("hello");
        let a2 = intern_symbol("hello");
        let b = intern_symbol("world");
        
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        
        assert_eq!(symbol_name(a1).unwrap(), "hello");
        assert_eq!(symbol_name(b).unwrap(), "world");
    }
    
    #[test]
    fn test_option_optimization() {
        // Option<SymbolId> should be same size as SymbolId due to NonZeroU32
        assert_eq!(
            std::mem::size_of::<Option<SymbolId>>(),
            std::mem::size_of::<SymbolId>()
        );
    }
}