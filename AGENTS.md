# Sea of Nodes Compiler IR

A Sea of Nodes compiler IR implementation in Rust with aggressive on-the-fly
optimization. The system avoids creating IR nodes whenever possible — peepholes,
constant folding, and value numbering happen during construction, not as
separate passes.

## Workflow Rules

- **No implementation without confirmation.** When we discuss, plan, or clarify
  something — if I say "let's plan X", ask for clarification about Y, or we're
  in a discussion phase — do not proceed to writing code until I explicitly
  say to go ahead (e.g., "implement it", "let's do it", "sounds good, build it").
  Discussion and planning are separate from implementation.
- **Always commit after implementing.** Once you finish writing code (and tests pass), commit immediately unless I explicitly tell you not to. Do not move on to another task or wait to be reminded — committing is part of finishing the implementation.
  - **Amend** the last commit if the change is obviously a fix to that commit (typo fix, missed edge case, test fix) or if I explicitly ask to amend.
  - Otherwise, create a **new commit**.
- **Update PLAN.md if needed after finishing work.** After implementing and committing, review whether the completed task should be marked done, reprioritized, or removed from the roadmap. Keep PLAN.md in sync with actual project state.

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
    ├── main.rs        # Module declarations, main entry point
    ├── node.rs        # Node, NodeId, NodeKind, node operations
    ├── builder.rs     # IRBuilder, SSA construction (Braun-style lazy Phis)
    ├── types.rs       # Type algebra (union, intersect, subtract, cast analysis)
    ├── constraints.rs # Generic RangeConstraint<T>, Bool/Float/Int/UInt constraints
    ├── compact_vec.rs # Multi-tier compact vector for node input storage
    ├── symbols.rs     # Thread-safe symbol interning with NonZeroU32
    └── tests/
        ├── mod.rs     # Test module declarations
        └── builder.rs # IRBuilder tests (constant folding, peepholes, value numbering)
```
