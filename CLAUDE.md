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

**Error Type Normalization**: The system uses `Error(T)` for typed errors with canonical normalization:
- `Error(Never)` → `Never`
- `Error(Error(T))` → `Error(T)` (unwraps nested errors)
- `Error(Union([A, B]))` → `Union([Error(A), Error(B)])` (distributes over unions)
- `Union([Error(A), Error(B)])` → `Error(Union([A, B]))` (merges adjacent errors)

This ensures only two canonical error forms exist:
1. Pure error: `Error(T)` where T contains no Error types
2. Mixed union: `Union([T1, Error(T2)])` where T1, T2 contain no Error types

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
├── main.rs          # Node definitions, IRBuilder, main compilation driver
├── types.rs         # Type system, union normalization, error handling
├── constraints.rs   # Generic constraint system, arithmetic operations
└── Cargo.toml       # Dependencies: ordered-float, num-traits
```

## Type System Implementation

### Union Normalization (`Type::make_union`)
1. **Flatten nested unions**: `Union([Union([A, B]), C])` → `[A, B, C]`
2. **Sort and deduplicate**: Ensures canonical ordering
3. **Handle special cases**: Any subsumes all, Never filtered out
4. **Merge adjacent ranges**: `[Int(1-5), Int(4-8)]` → `Int(1-8)`
5. **Merge adjacent errors**: `[Error(A), Error(B)]` → `Error(Union([A, B]))`
6. **Collapse single elements**: `Union([T])` → `T`

### Error Distribution (`Type::normalize`)
- `Error(Union([A, B]))` gets distributed to `Union([Error(A), Error(B)])`
- Then union merging consolidates back to `Error(Union([A, B]))`
- This flattens nested errors: `Error(Union([A, Error(B)]))` → `Error(Union([A, B]))`

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
cargo check          # Type check
cargo test           # Run all tests
cargo run            # Build and run
```

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

### Future Extensibility
The generic constraint system allows easy addition of new constraint types by implementing the `ConstraintValue` trait. The error normalization system supports rich error types with full type information.