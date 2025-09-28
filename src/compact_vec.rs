use std::alloc::{Layout, alloc, dealloc, realloc};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StorageType {
    InlineU8 = 0,
    InlineU16 = 1,
    InlineU32 = 2,
    Heap = 3,
}

pub trait CompactVecState: Copy + Clone {
    fn new(storage_type: StorageType, len: usize) -> Self;
    fn len(self) -> usize;
    fn storage_type(self) -> StorageType;

    #[inline]
    fn new_empty() -> Self {
        Self::new(StorageType::InlineU8, 0)
    }

    #[inline]
    fn set_len(&mut self, len: usize) {
        *self = Self::new(self.storage_type(), len);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactVecStateU8(u8);

impl CompactVecState for CompactVecStateU8 {
    #[inline]
    fn new(storage_type: StorageType, len: usize) -> Self {
        assert!(
            len <= 63,
            "Length {} exceeds 6-bit capacity for u8 state",
            len
        );
        Self(((storage_type as u8) << 6) | (len as u8))
    }

    #[inline]
    fn len(self) -> usize {
        (self.0 & 0b0011_1111) as usize
    }

    #[inline]
    fn storage_type(self) -> StorageType {
        unsafe { std::mem::transmute((self.0 & (0b11 << 6)) >> 6) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactVecStateU16(u16);

impl CompactVecState for CompactVecStateU16 {
    #[inline]
    fn new(storage_type: StorageType, len: usize) -> Self {
        assert!(
            len <= 16383,
            "Length {} exceeds 14-bit capacity for u16 state",
            len
        );
        Self(((storage_type as u16) << 14) | (len as u16))
    }

    #[inline]
    fn len(self) -> usize {
        (self.0 & 0b0011_1111_1111_1111) as usize
    }

    #[inline]
    fn storage_type(self) -> StorageType {
        unsafe { std::mem::transmute(((self.0 & (0b11 << 14)) >> 14) as u8) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactVecStateU32(u32);

impl CompactVecState for CompactVecStateU32 {
    #[inline]
    fn new(storage_type: StorageType, len: usize) -> Self {
        assert!(
            len <= 1073741823,
            "Length {} exceeds 30-bit capacity for u32 state",
            len
        );
        Self(((storage_type as u32) << 30) | (len as u32))
    }

    #[inline]
    fn len(self) -> usize {
        (self.0 & 0x3FFFFFFF) as usize
    }

    #[inline]
    fn storage_type(self) -> StorageType {
        unsafe { std::mem::transmute(((self.0 & (0b11 << 30)) >> 30) as u8) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactVecStateU64(u64);

impl CompactVecState for CompactVecStateU64 {
    #[inline]
    fn new(storage_type: StorageType, len: usize) -> Self {
        assert!(
            len <= 4611686018427387903,
            "Length {} exceeds 62-bit capacity for u64 state",
            len
        );
        Self(((storage_type as u64) << 62) | (len as u64))
    }

    #[inline]
    fn len(self) -> usize {
        (self.0 & 0x3FFFFFFFFFFFFFFF) as usize
    }

    #[inline]
    fn storage_type(self) -> StorageType {
        unsafe { std::mem::transmute(((self.0 & (0b11 << 62)) >> 62) as u8) }
    }
}

#[derive(Copy, Clone)]
pub union CompactVecData<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> {
    inline_u8: [u8; MAX_U8],
    inline_u16: [u16; MAX_U16],
    inline_u32: [u32; MAX_U32],
    heap_ptr: *mut u32,
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> fmt::Debug
    for CompactVecData<MAX_U8, MAX_U16, MAX_U32>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompactVec").finish_non_exhaustive()
    }
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> Default
    for CompactVecData<MAX_U8, MAX_U16, MAX_U32>
{
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize>
    CompactVecData<MAX_U8, MAX_U16, MAX_U32>
{
    #[inline]
    pub fn get(&self, state: &impl CompactVecState, index: usize) -> u32 {
        assert!(index < state.len());

        unsafe {
            match state.storage_type() {
                StorageType::InlineU8 => self.inline_u8[index] as u32,
                StorageType::InlineU16 => self.inline_u16[index] as u32,
                StorageType::InlineU32 => self.inline_u32[index],
                StorageType::Heap => {
                    assert!(!self.heap_ptr.is_null());
                    // Data starts at offset 1, capacity at offset 0
                    *self.heap_ptr.add(1 + index)
                }
            }
        }
    }

    pub fn set_all(&mut self, state: &mut impl CompactVecState, items: &[u32]) {
        if items.is_empty() {
            *state = <_>::new_empty();
            return;
        }

        let max_value = *items.iter().max().unwrap();

        // Choose storage and set items in one pass
        if items.len() <= MAX_U8 && max_value <= u8::MAX as u32 {
            // InlineU8
            *state = <_>::new(StorageType::InlineU8, items.len());
            unsafe {
                for (i, &item) in items.iter().enumerate() {
                    self.inline_u8[i] = item as u8;
                }
            }
        } else if items.len() <= MAX_U16 && max_value <= u16::MAX as u32 {
            // InlineU16
            *state = <_>::new(StorageType::InlineU16, items.len());
            unsafe {
                for (i, &item) in items.iter().enumerate() {
                    self.inline_u16[i] = item as u16;
                }
            }
        } else if items.len() <= MAX_U32 {
            // InlineU32
            *state = <_>::new(StorageType::InlineU32, items.len());
            unsafe {
                for (i, &item) in items.iter().enumerate() {
                    self.inline_u32[i] = item;
                }
            }
        } else {
            // Heap: allocate capacity + 1 for storing capacity at [0]
            let capacity = items.len().max(4); // Minimum capacity of 4
            *state = <_>::new(StorageType::Heap, items.len());
            let layout = Layout::array::<u32>(capacity + 1).unwrap();
            unsafe {
                self.heap_ptr = alloc(layout) as *mut u32;
                assert!(!self.heap_ptr.is_null());
                // Store capacity at offset 0
                *self.heap_ptr = capacity as u32;
                // Store data starting at offset 1
                for (i, &item) in items.iter().enumerate() {
                    *self.heap_ptr.add(1 + i) = item;
                }
            }
        }
    }

    pub fn set(&mut self, state: &mut impl CompactVecState, index: usize, item: u32) {
        assert!(index < state.len());

        unsafe {
            match state.storage_type() {
                StorageType::InlineU8 => {
                    if item <= u8::MAX as u32 {
                        self.inline_u8[index] = item as u8;
                    } else {
                        self.promote_and_set(state, index, item);
                    }
                }
                StorageType::InlineU16 => {
                    if item <= u16::MAX as u32 {
                        self.inline_u16[index] = item as u16;
                    } else {
                        self.promote_and_set(state, index, item);
                    }
                }
                StorageType::InlineU32 => {
                    self.inline_u32[index] = item;
                }
                StorageType::Heap => {
                    assert!(!self.heap_ptr.is_null());
                    // Data starts at offset 1
                    *self.heap_ptr.add(1 + index) = item;
                }
            }
        }
    }

    pub fn push(&mut self, state: &mut impl CompactVecState, item: u32) {
        let current_len = state.len();

        unsafe {
            match state.storage_type() {
                StorageType::InlineU8 => {
                    if current_len < MAX_U8 && item <= u8::MAX as u32 {
                        self.inline_u8[current_len] = item as u8;
                        state.set_len(current_len + 1);
                    } else {
                        self.promote_and_set(state, current_len, item);
                    }
                }
                StorageType::InlineU16 => {
                    if current_len < MAX_U16 && item <= u16::MAX as u32 {
                        self.inline_u16[current_len] = item as u16;
                        state.set_len(current_len + 1);
                    } else {
                        self.promote_and_set(state, current_len, item);
                    }
                }
                StorageType::InlineU32 => {
                    if current_len < MAX_U32 {
                        self.inline_u32[current_len] = item;
                        state.set_len(current_len + 1);
                    } else {
                        self.promote_and_set(state, current_len, item);
                    }
                }
                StorageType::Heap => {
                    let current_len = state.len();
                    let capacity = *self.heap_ptr as usize;

                    if current_len < capacity {
                        // Space available, just add the item
                        *self.heap_ptr.add(1 + current_len) = item;
                        state.set_len(current_len + 1);
                    } else {
                        // Need to grow: double capacity
                        let new_capacity = capacity * 2;
                        let old_layout = Layout::array::<u32>(capacity + 1).unwrap();
                        let new_layout = Layout::array::<u32>(new_capacity + 1).unwrap();

                        let new_ptr =
                            realloc(self.heap_ptr as *mut u8, old_layout, new_layout.size())
                                as *mut u32;
                        assert!(!new_ptr.is_null());

                        // Update capacity
                        *new_ptr = new_capacity as u32;
                        // Add new item
                        *new_ptr.add(1 + current_len) = item;

                        self.heap_ptr = new_ptr;
                        state.set_len(current_len + 1);
                    }
                }
            }
        }
    }

    fn promote_and_set(&mut self, state: &mut impl CompactVecState, index: usize, item: u32) {
        let current_len = state.len();
        let new_len = if index == current_len {
            current_len + 1
        } else {
            assert!(index < current_len);
            current_len
        };
        let mut items = [0u32; 64]; // Large fixed array, only use what we need
        for i in 0..current_len {
            items[i] = self.get(state, i);
        }
        items[index] = item;
        self.set_all(state, &items[..new_len]);
    }

    #[inline]
    pub fn clone_heap(&self, state: &impl CompactVecState) -> Self {
        if state.storage_type() == StorageType::Heap {
            unsafe {
                assert!(!self.heap_ptr.is_null());
                let capacity = *self.heap_ptr as usize;
                let len = state.len();
                let layout = Layout::array::<u32>(capacity + 1).unwrap();

                let new_ptr = alloc(layout) as *mut u32;
                assert!(!new_ptr.is_null());

                // Copy capacity + data
                std::ptr::copy_nonoverlapping(self.heap_ptr, new_ptr, len + 1);

                Self { heap_ptr: new_ptr }
            }
        } else {
            // For inline storage, just copy the union
            *self
        }
    }

    #[inline]
    pub fn drop_heap(&mut self, state: &impl CompactVecState) {
        if state.storage_type() == StorageType::Heap {
            unsafe {
                assert!(!self.heap_ptr.is_null());
                let capacity = *self.heap_ptr as usize;
                let layout = Layout::array::<u32>(capacity + 1).unwrap();
                dealloc(self.heap_ptr as *mut u8, layout);
            }
        }
    }
}

pub struct CompactVec<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> {
    state: CompactVecStateU64,
    data: CompactVecData<MAX_U8, MAX_U16, MAX_U32>,
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> Clone
    for CompactVec<MAX_U8, MAX_U16, MAX_U32>
{
    fn clone(&self) -> Self {
        Self {
            state: self.state,
            data: self.data.clone_heap(&self.state),
        }
    }
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> Drop
    for CompactVec<MAX_U8, MAX_U16, MAX_U32>
{
    fn drop(&mut self) {
        self.data.drop_heap(&self.state);
    }
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize> fmt::Debug
    for CompactVec<MAX_U8, MAX_U16, MAX_U32>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompactVec")
            .field("len", &self.len())
            .field("storage_type", &self.state.storage_type())
            .finish()
    }
}

impl<const MAX_U8: usize, const MAX_U16: usize, const MAX_U32: usize>
    CompactVec<MAX_U8, MAX_U16, MAX_U32>
{
    #[inline]
    pub fn new() -> Self {
        Self {
            state: CompactVecStateU64::new_empty(),
            data: unsafe { std::mem::zeroed() },
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.state.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> u32 {
        self.data.get(&self.state, index)
    }

    #[inline]
    pub fn set_all(&mut self, items: &[u32]) {
        self.data.set_all(&mut self.state, items)
    }

    #[inline]
    pub fn set(&mut self, index: usize, item: u32) {
        self.data.set(&mut self.state, index, item)
    }

    #[inline]
    pub fn push(&mut self, item: u32) {
        self.data.push(&mut self.state, item)
    }

    #[inline]
    pub fn storage_type(&self) -> StorageType {
        self.state.storage_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestVec = CompactVecData<4, 2, 1>; // Small sizes for testing: 4 u8s, 2 u16s, 1 u32

    #[test]
    fn test_state_u8_basic() {
        let mut state = CompactVecStateU8::new(StorageType::InlineU8, 42);
        assert_eq!(state.len(), 42);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        state.set_len(50);
        assert_eq!(state.len(), 50);
        assert_eq!(state.storage_type(), StorageType::InlineU8);
    }

    #[test]
    fn test_state_u8_max_len() {
        let state = CompactVecStateU8::new(StorageType::InlineU8, 63); // 6 bits = max 63
        assert_eq!(state.len(), 63);
    }

    #[test]
    #[should_panic]
    fn test_state_u8_overflow() {
        let _state = CompactVecStateU8::new(StorageType::InlineU8, 64); // Should overflow 6 bits
    }

    #[test]
    fn test_state_u16_basic() {
        let mut state = CompactVecStateU16::new(StorageType::InlineU16, 1000);
        assert_eq!(state.len(), 1000);
        assert_eq!(state.storage_type(), StorageType::InlineU16);

        state.set_len(16383); // Max for 14 bits
        assert_eq!(state.len(), 16383);
    }

    #[test]
    #[should_panic]
    fn test_state_u16_overflow() {
        let _state = CompactVecStateU16::new(StorageType::InlineU16, 16384); // Should overflow 14 bits
    }

    #[test]
    fn test_state_u32_basic() {
        let state = CompactVecStateU32::new(StorageType::Heap, 1000000000);
        assert_eq!(state.len(), 1000000000);
        assert_eq!(state.storage_type(), StorageType::Heap);
    }

    #[test]
    fn test_state_u64_basic() {
        let state = CompactVecStateU64::new(StorageType::Heap, 1000000000000);
        assert_eq!(state.len(), 1000000000000);
        assert_eq!(state.storage_type(), StorageType::Heap);
    }

    #[test]
    fn test_all_storage_types() {
        for &storage_type in &[
            StorageType::InlineU8,
            StorageType::InlineU16,
            StorageType::InlineU32,
            StorageType::Heap,
        ] {
            let state = CompactVecStateU16::new(storage_type, 42);
            assert_eq!(state.storage_type(), storage_type);
            assert_eq!(state.len(), 42);
        }
    }

    #[test]
    fn test_empty_vec() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        assert_eq!(state.len(), 0);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        vec.set_all(&mut state, &[]);
        assert_eq!(state.len(), 0);
        assert_eq!(state.storage_type(), StorageType::InlineU8);
    }

    #[test]
    fn test_inline_u8_storage() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Values that fit in u8 and length <= MAX_U8
        let items = [10, 20, 255, 100];
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::InlineU8);
        assert_eq!(state.len(), 4);

        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }
    }

    #[test]
    fn test_inline_u8_to_u16_promotion_by_value() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Start with u8 values
        vec.set_all(&mut state, &[10, 20]);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Value too large for u8 should promote to u16 (2 items <= MAX_U16=2)
        let items = [10, 300]; // 300 > u8::MAX, 2 items <= MAX_U16=2
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::InlineU16);
        assert_eq!(state.len(), 2);

        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }
    }

    #[test]
    fn test_inline_u8_to_u16_promotion_by_count() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Too many items for u8 storage (MAX_U8 = 4)
        let items = [1, 2, 3, 4, 5]; // 5 items > MAX_U8=4, should go to heap since 5 > MAX_U16=2
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::Heap);
        assert_eq!(state.len(), 5);

        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }

        vec.drop_heap(&state);
    }

    #[test]
    fn test_inline_u16_to_u32_promotion() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Single value too large for u16, should promote to u32 (1 item <= MAX_U32=1)
        let items = [65536]; // > u16::MAX, 1 item = MAX_U32=1
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::InlineU32);
        assert_eq!(state.len(), 1);
        assert_eq!(vec.get(&state, 0), 65536);
    }

    #[test]
    fn test_inline_u32_to_heap_by_count() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Two large values, exceeds MAX_U32=1, should go to heap
        let items = [65536, 65537]; // > u16::MAX, 2 items > MAX_U32=1
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::Heap);
        assert_eq!(state.len(), 2);

        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }

        vec.drop_heap(&state);
    }

    #[test]
    fn test_inline_u32_to_heap_promotion() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // For TestVec<4,3,2>, small values will be InlineU8 first
        let items = [1, 2, 3]; // 3 items <= MAX_U8=4, small values
        vec.set_all(&mut state, &items);

        assert_eq!(state.storage_type(), StorageType::InlineU8);
        assert_eq!(state.len(), 3);

        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }
    }

    #[test]
    fn test_set_with_promotion() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Start with u8 storage (2 items, fits in both u8 and u16)
        vec.set_all(&mut state, &[10, 20]);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Setting a large value should promote to u16 (2 items = MAX_U16=2)
        vec.set(&mut state, 1, 300); // 300 > u8::MAX
        assert_eq!(state.storage_type(), StorageType::InlineU16);
        assert_eq!(vec.get(&state, 0), 10);
        assert_eq!(vec.get(&state, 1), 300);
    }

    #[test]
    fn test_push_with_promotion() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Start with u8 storage (2 items fits in both u8 and u16)
        vec.set_all(&mut state, &[10, 20]);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Push a large value should promote to u16 (3 items > MAX_U16=2, so goes to heap)
        vec.push(&mut state, 400);
        assert_eq!(state.storage_type(), StorageType::Heap);
        assert_eq!(state.len(), 3);
        assert_eq!(vec.get(&state, 2), 400);

        vec.drop_heap(&state);
    }

    #[test]
    fn test_push_capacity_growth() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Force heap storage by adding too many items for inline storage
        vec.set_all(&mut state, &[1, 2, 3, 4, 5]); // 5 > MAX_U16=2, goes to heap
        assert_eq!(state.storage_type(), StorageType::Heap);

        // Push many items to test capacity growth
        for i in 6u32..=10 {
            vec.push(&mut state, i);
            assert_eq!(state.len(), i as usize);
            assert_eq!(vec.get(&state, i as usize - 1), i);
        }

        vec.drop_heap(&state);
    }

    #[test]
    fn test_heap_capacity_storage() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Create heap storage
        let items: Vec<u32> = (1..=20).collect();
        vec.set_all(&mut state, &items);
        assert_eq!(state.storage_type(), StorageType::Heap);

        // Verify capacity is stored at heap_ptr[0]
        unsafe {
            let capacity = *vec.heap_ptr;
            assert!(capacity >= 20); // Should be at least the number of items
            assert!(capacity >= 4); // Minimum capacity
        }

        // Verify data is stored starting at heap_ptr[1]
        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec.get(&state, i), expected);
        }

        vec.drop_heap(&state);
    }

    #[test]
    fn test_clone_heap() {
        let mut vec1 = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Create heap storage
        let items = [100, 200, 300, 400, 500];
        vec1.set_all(&mut state, &items);
        assert_eq!(state.storage_type(), StorageType::Heap);

        // Clone the heap
        let vec2 = vec1.clone_heap(&state);

        // Verify both vectors have the same data
        for (i, &expected) in items.iter().enumerate() {
            assert_eq!(vec1.get(&state, i), expected);
            assert_eq!(vec2.get(&state, i), expected);
        }

        // Verify they have different heap pointers
        unsafe {
            assert_ne!(vec1.heap_ptr, vec2.heap_ptr);
        }

        // Clean up both heaps
        vec1.drop_heap(&state);
        let mut vec2_mut = vec2;
        vec2_mut.drop_heap(&state);
    }

    #[test]
    fn test_clone_inline() {
        let mut vec1 = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Create inline storage
        vec1.set_all(&mut state, &[1, 2, 3]);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Clone should just copy the union
        let vec2 = vec1.clone_heap(&state);

        for i in 0..3 {
            assert_eq!(vec1.get(&state, i), vec2.get(&state, i));
        }
    }

    #[test]
    fn test_comprehensive_transitions() {
        // Use larger sizes for this test to see all transitions
        type LargeTestVec = CompactVecData<8, 6, 4>;
        let mut vec = LargeTestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        // Start empty (InlineU8)
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Add small values (stay InlineU8)
        vec.set_all(&mut state, &[1, 2]);
        assert_eq!(state.storage_type(), StorageType::InlineU8);

        // Add large value (promote to InlineU16)
        vec.set_all(&mut state, &[1, 2, 300]);
        assert_eq!(state.storage_type(), StorageType::InlineU16);

        // Add very large value (promote to InlineU32)
        vec.set_all(&mut state, &[1, 2, 300, 70000]);
        assert_eq!(state.storage_type(), StorageType::InlineU32);

        // Add too many items (promote to Heap)
        vec.set_all(&mut state, &[1, 2, 300, 70000, 5]);
        assert_eq!(state.storage_type(), StorageType::Heap);

        // Verify all values
        assert_eq!(vec.get(&state, 0), 1);
        assert_eq!(vec.get(&state, 1), 2);
        assert_eq!(vec.get(&state, 2), 300);
        assert_eq!(vec.get(&state, 3), 70000);
        assert_eq!(vec.get(&state, 4), 5);

        vec.drop_heap(&state);
    }

    #[test]
    fn test_bounds_checking() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();

        vec.set_all(&mut state, &[1, 2, 3]);

        // Valid indices should work
        assert_eq!(vec.get(&state, 0), 1);
        assert_eq!(vec.get(&state, 2), 3);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();
        vec.set_all(&mut state, &[1, 2, 3]);
        vec.get(&state, 3); // Should panic
    }

    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut vec = TestVec::default();
        let mut state = CompactVecStateU16::new_empty();
        vec.set_all(&mut state, &[1, 2, 3]);
        vec.set(&mut state, 3, 100); // Should panic
    }

    #[test]
    fn test_combined_compact_vec() {
        let mut vec = CompactVec::<4, 3, 2>::new();

        // Test basic operations
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.storage_type(), StorageType::InlineU8);

        // Test set_all
        vec.set_all(&[10, 20, 30]);
        assert_eq!(vec.len(), 3);
        assert!(!vec.is_empty());
        assert_eq!(vec.get(0), 10);
        assert_eq!(vec.get(1), 20);
        assert_eq!(vec.get(2), 30);

        // Test push
        vec.push(40);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(3), 40);

        // Test set
        vec.set(1, 200);
        assert_eq!(vec.get(1), 200);

        // Test clone
        let vec2 = vec.clone();
        assert_eq!(vec2.len(), vec.len());
        for i in 0..vec.len() {
            assert_eq!(vec2.get(i), vec.get(i));
        }
    }

    #[test]
    fn test_different_state_types() {
        // Test that different state types work with the same data
        let mut vec = TestVec::default();

        // Test with U8 state (limited length)
        let mut state_u8 = CompactVecStateU8::new_empty();
        vec.set_all(&mut state_u8, &[1, 2, 3]);
        assert_eq!(state_u8.len(), 3);
        assert_eq!(vec.get(&state_u8, 1), 2);

        // Test with U16 state (more length)
        let mut state_u16 = CompactVecStateU16::new_empty();
        vec.set_all(&mut state_u16, &[10, 20, 30, 40]);
        assert_eq!(state_u16.len(), 4);
        assert_eq!(vec.get(&state_u16, 3), 40);

        // Test with U64 state (massive length capacity)
        let mut state_u64 = CompactVecStateU64::new_empty();
        vec.set_all(&mut state_u64, &[100, 200]);
        assert_eq!(state_u64.len(), 2);
        assert_eq!(vec.get(&state_u64, 0), 100);
    }
}
