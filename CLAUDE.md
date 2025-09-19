# Sea of Nodes Compiler IR - CLAUDE.md

## Project Overview

This is a Sea of Nodes compiler IR implementation in Rust with aggressive on-the-fly optimization. The system is designed to avoid IR node creation through constructor peepholes, constant folding, and value numbering.

**Current Status**: 3000+ lines of Rust code implementing a complete type system with function types, cast analysis, constraint arithmetic, and comprehensive union/error normalization.

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
├── main.rs          # Node definitions (768 lines), IRBuilder, cast functionality
├── types.rs         # Type system (1675 lines), union/error flattening, function types
├── constraints.rs   # Generic constraint system (524 lines), arithmetic operations
├── symbols.rs       # Thread-safe symbol interning (89 lines) with NonZeroU32 optimization
└── Cargo.toml       # Dependencies: ordered-float, num-traits
```

**Core Modules**:
- **Node System**: 48-byte nodes with embedded types, IRError enum, NodeKind with memory/control operations
- **Type System**: Complete type algebra with Function types, Data types, cast analysis (CastKind)
- **Constraints**: ConstraintValue trait, IntPrim/UIntPrim/FloatPrim enums, checked arithmetic
- **Symbols**: Thread-safe interning with NonZeroU32 space optimization

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

### Function Types (`Type::Fun`)
- **Structural parameter equality**: Function types equal if return type and parameter types match (names ignored)
- **FunParam structure**: Contains name (for errors/docs) and type, with custom equality ignoring names
- **Arc<FunInfo>**: Efficient sharing of function signatures across multiple nodes
- **Complete integration**: Functions participate in union/intersection/subtraction type algebra

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

**CRITICAL**: Always use `cargo test` exclusively for all development work.

```bash
cargo test           # ONLY command to use - runs all tests, builds, and validates
```

**⚠️ IMPORTANT REQUIREMENTS**:
- **NEVER** use `cargo check`, `cargo build`, `cargo run`, or individual test commands
- **NEVER** use `cargo test specific_test_name` - always run the full test suite
- **ALWAYS** use `cargo test` for any code validation, building, or testing
- The full test suite is optimized to run everything at once efficiently
- Individual commands are slower and may miss critical integration issues

**Why cargo test only**: The comprehensive test suite validates the entire type system, constraint arithmetic, and IR construction in one optimized pass. Running partial tests or builds can miss subtle interactions between components.

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
**Complete coverage**: Handles all primitive types (Bool, Int, UInt, Float) with constraint propagation

### Symbol Interning System
**Thread-safe design** with `RwLock<SymbolTable>` and `OnceLock` for global state
**Space optimization**: `SymbolId(NonZeroU32)` enables `Option<SymbolId>` same size as `SymbolId`  
**Performance**: Read-preferring locks for efficient concurrent symbol lookup

### Node System Architecture
**IRError enum**: Comprehensive error handling for TypeMismatch, IntegerOverflow, DivisionByZero, InvalidPrimitiveCoercion
**NodeKind variants**: Complete set including Unreachable, Interned, Param, Const, Phi, control nodes (Entry, If, Then, Else, Region, Loop), memory nodes (Memory, New, Load, Store), arithmetic ops (Add, Sub, Mul, Div, Neg, Not), and cast operations (StaticCast, DynamicCast)
**48-byte Node structure**: Efficient union-based NodeData with inputs array, parameter indices, and interned node IDs
**Value numbering ready**: Hash and equality implementations for aggressive IR deduplication

### Current Implementation Status
- ✅ **Complete type system** with union normalization, error distribution, function types
- ✅ **Full constraint arithmetic** with overflow detection and type coercion
- ✅ **Three-way cast analysis** covering all primitive type combinations
- ✅ **Thread-safe symbol interning** with space-optimized NonZeroU32 IDs
- ✅ **IR node definitions** with memory, control, and arithmetic operations
- 🚧 **IRBuilder implementation** for aggressive on-the-fly optimization (in progress)

### Future Extensibility
The generic constraint system allows easy addition of new constraint types by implementing the `ConstraintValue` trait. The product type system supports future syntax extensions like `data Point(x: i32, y: i32)` and anonymous structural types.