# Sea of Nodes Compiler IR - CLAUDE.md

## Project Overview

This is a Sea of Nodes compiler IR implementation in Rust with aggressive on-the-fly optimization. The system is designed to avoid IR node creation through constructor peepholes, constant folding, and value numbering.

## Key Architecture Decisions

### Type System Design

**First-Level By-Value Storage**: Types are stored directly in nodes rather than through indirection (Arc/TypeId). This increases node size to 48 bytes but provides:
- Zero allocation for common types (I32, Bool, etc.)
- Fast type access critical for constructor peepholes and constant folding
- No hash table lookups for simple type operations
- Direct type inspection during optimization

**Union Flattening for Runtime Efficiency**: The system flattens errors to individual union members:
- `Error(Union([A, B]))` → `Union([Error(A), Error(B)])` (distributes immediately)
- No nested `Error(Union(...))` structures exist
- Errors naturally sort to the tail of unions due to derived `Ord`
- Enables efficient runtime representation: `tag + data` with `tag >= first_error_index` for error checking
- Simple sequential tag indexing for all union members

### Generic Constraint System

**RangeConstraint<T>**: Unified constraint implementation using the `ConstraintValue` trait:
- Abstracts over i64/u64 storage types
- Preserves primitive type information (IntPrim/UIntPrim)
- Handles untyped constant coercion (i64 constants can become i8/i16/i32)
- Supports checked arithmetic with overflow detection

**Type Aliases**: `IntConstraint = RangeConstraint<i64>`, `UIntConstraint = RangeConstraint<u64>`

### Optimization Strategy

**On-the-Fly Optimization**: Most optimizations happen during IR construction:
- Constructor peepholes catch simple patterns
- Constant folding eliminates temporary nodes
- Value numbering deduplicates identical operations
- Result: 20x fewer nodes than traditional compilers

**Comptime Evaluation**: Designed for compile-time evaluation (like Zig comptime):
- Recursive compilation drives IR generation
- Comptime loops execute directly rather than generating IR
- All operations go through IRBuilder for consistent constant folding
- Pure comptime code generates zero IR nodes

## Project Structure

```
src/
├── main.rs          # Node definitions, IRBuilder, cast functionality
├── types.rs         # Type system, union/error flattening, product types  
├── constraints.rs   # Generic constraint system, arithmetic operations
├── symbols.rs       # Thread-safe symbol interning with NonZeroU32 optimization
└── Cargo.toml       # Dependencies: ordered-float, num-traits
```

## Type System Implementation

### Union Normalization (`Type::make_union`)
1. **Flatten nested unions**: `Union([Union([A, B]), C])` → `[A, B, C]`
2. **Distribute error unions**: `Error(Union([A, B]))` → `[Error(A), Error(B)]`
3. **Sort and deduplicate**: Ensures canonical ordering (errors naturally sort to tail)
4. **Handle special cases**: Any subsumes all, Never filtered out
5. **Merge adjacent ranges**: `[Int(1-5), Int(4-8)]` → `Int(1-8)`
6. **No error merging**: Prevents nested `Error(Union(...))` structures
7. **Collapse single elements**: `Union([T])` → `T`

### Product Types (`Type::Data`)
- **Structural nominal typing**: Same tag + same fields = equal types
- **Symbol interning**: Field names stored as `SymbolId` with `NonZeroU32` optimization
- **Unified representation**: Both tuples and records use `DataField` array
- **Field naming**: Tuples use "0", "1", etc.; records use actual field names
- **Helper constructors**: `make_tuple`, `make_named_tuple`, `make_record`, `make_named_record`

## Memory and Performance Characteristics

### Node Size Trade-offs
- **48-byte nodes** with by-value types vs **32-byte nodes** with TypeId indirection
- Choice: By-value types for zero indirection cost
- Justified by aggressive node reduction (20x fewer nodes created)
- Total memory impact minimal, type access performance critical

### Arc vs Box for Nested Types
- **Arc<Type>**: Cheap cloning (O(1)), shared ownership, refcount overhead
- **Box<Type>**: Simple ownership, expensive cloning (O(n)), no sharing
- Choice: Arc for comptime evaluation with heavy temporary cloning
- Future: Could switch to TypeId interning while keeping first-level by-value

## Testing Strategy

Comprehensive test coverage for:
- Union normalization (flattening, merging, deduplication)
- Error normalization (distribution, unwrapping, merging)
- Range constraint arithmetic (overflow handling, type coercion)
- Type algebra operations (intersection, subtraction)

## Build Commands

```bash
cargo test           # Run all tests (preferred - tests everything at once)
cargo run            # Build and run
```

**Note**: Only run `cargo test` (not `cargo check` or specific tests) as it's faster to test everything at once.

## Implementation Notes

### Union Type Algebra
The system implements a complete type algebra with:
- **Union**: `A ∪ B` (make_union)
- **Intersection**: `A ∩ B` (intersect) 
- **Subtraction**: `A - B` (subtract)

Operations maintain canonical forms and handle edge cases (Never, Any, overlapping ranges).

### Constraint Arithmetic
All arithmetic operations (add, sub, mul, div) include:
- Overflow detection with checked arithmetic
- Primitive type resolution and coercion
- Range propagation for optimization
- Error handling for provable overflow vs runtime cases

### Cast System
**Three-way cast analysis** with `CastKind` enum:
- **Static**: Guaranteed safe, no runtime check (e.g., widening with compatible ranges)
- **Dynamic**: Requires runtime type check (e.g., narrowing or cross-type casts)  
- **Invalid**: Definitely impossible (e.g., disjoint ranges)

**Range-based analysis**: Uses `CommonRange` (i128) for unified signed/unsigned comparison
**Cross-type support**: Signed ↔ unsigned casts based on range overlap/containment

### Symbol Interning System
**Thread-safe design** with `RwLock<SymbolTable>` and `OnceLock` for global state
**Space optimization**: `SymbolId(NonZeroU32)` enables `Option<SymbolId>` same size as `SymbolId`  
**Performance**: Read-preferring locks for efficient concurrent symbol lookup

### Future Extensibility
The generic constraint system allows easy addition of new constraint types by implementing the `ConstraintValue` trait. The product type system supports future syntax extensions like `data Point(x: i32, y: i32)` and anonymous structural types.