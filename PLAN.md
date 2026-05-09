# PLAN — Sea of Nodes Compiler IR

Living document of goals and priorities. Items can be coarse or fine-grained.
When something new comes up, add it here.

---

## Current Priorities

### P0 ✅ — Memory SSA
Memory nodes (`New`, `Load`, `Store`) now have builder methods:
`create_new`, `create_load`, `create_store`. Memory is tracked as a first-class
SSA value — the same Braun-style algorithm used for variables handles memory
reaching definitions at control merges, with automatic Phi insertion and
trivial Phi elimination. Memory-producing ops (`New`, `Store`) auto-update
the memory chain. `Load` is "amphibious" — it reads from memory without
producing a new memory state.

### P1 — Graph visualization
A `Display` or Debug that emits Graphviz `.dot` output. Debugging graph IRs
by reading node arrays is painful. Visualization accelerates all future work.

### P1 — Memory peephole optimizations
Fold `Load` immediately after `Store` to same ptr → return stored value.
Fold `Load` from `New` before any `Store` → return zero/undef initial value.
Eliminate dead `Store` chains (store to ptr that is never subsequently loaded).
These are the memory equivalent of the arithmetic peepholes we already have.

### P2 — Graph visualization
A `Display` or Debug that emits Graphviz `.dot` output. Debugging graph IRs
by reading node arrays is painful. Visualization accelerates all future work.

### P3 — End-to-end demo
A tiny expression language + parser that exercises the full pipeline
(syntax → IR → constant folding → SSA → memory ops). Makes the project
tangible and catch integration bugs.

---

## Icebox

- **Loop optimization**: strength reduction, invariant hoisting
- **Dead node GC**: mark-sweep to reclaim space, clean up `interned_nodes`
- **Comptime evaluation**: compile-time execution (like Zig comptime)
- **Function types**: function call IR nodes, return/param wiring
- **Cross-type arithmetic**: automatic int/uint/float coercion rules
- **Better type inference on Phis**: union of operand types is correct but
  coarse — could be narrowed with lattice meet
- **Freelist / `kill_node`**: deferred — NodeId stability is more important
  than memory reuse at this stage