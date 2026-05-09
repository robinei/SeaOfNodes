# Sea of Nodes Compiler IR

A Sea of Nodes compiler IR implementation in Rust with aggressive on-the-fly
optimization. The system avoids creating IR nodes whenever possible — peepholes,
constant folding, and value numbering happen during construction, not as
separate passes.

## What We Value

- **Compact representations, cache efficiency**: Avoid pointer chasing.
  Prefer integer handles (NodeId, VarId, SymbolId) over pointers. Store
  objects in dense vectors, not behind individual allocations. This applies
  to everything — nodes, types, symbols.
- **Optimize during construction, not after**: Peepholes, constant folding,
  and value numbering fire as nodes are created. Lazy SSA (Braun) places
  Phis only at joins with different reaching definitions and eliminates
  trivial ones immediately. No separate optimization passes.

## Build & Test

```bash
cargo test    # ONLY command — never cargo check/build/run individually
              # Full suite catches cross-module integration issues
```

See [`PLAN.md`](./PLAN.md) for the prioritized roadmap.

## File Map

```
├── AGENTS.md          # This file
├── PLAN.md            # Prioritized goals
├── Cargo.toml
└── src/
    ├── main.rs        # Node, IRBuilder, SSA construction, all tests
    ├── types.rs       # Type algebra (union, intersect, subtract, cast analysis)
    ├── constraints.rs # Generic RangeConstraint<T>, Bool/Float/Int/UInt constraints
    ├── compact_vec.rs # Multi-tier compact vector for node input storage
    └── symbols.rs     # Thread-safe symbol interning with NonZeroU32
```
