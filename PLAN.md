# PLAN — Sea of Nodes Compiler IR

Living document of goals and priorities. Items can be coarse or fine-grained.
When something new comes up, add it here.

---

## Current Priorities

### P0 — Memory SSA
Memory nodes (`New`, `Load`, `Store`) are defined as `NodeKind` variants
but have zero builder methods. Memory is a first-class value in Sea of Nodes
that flows through the graph like any other value. Without this, we can't
compile programs with mutable state.

Blockers: none — type system already has `Type::Memory`, control flow and
SSA construction are ready.

### P1 — Graph visualization
A `Display` or Debug that emits Graphviz `.dot` output. Debugging graph IRs
by reading node arrays is painful. Visualization accelerates all future work.

### P2 — End-to-end demo
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