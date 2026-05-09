# PLAN — Sea of Nodes Compiler IR

Living document of goals and priorities. Items can be coarse or fine-grained.
When something new comes up, add it here.

---

## Current Priorities

### Graph visualization
A `Display` or Debug that emits Graphviz `.dot` output. Debugging graph IRs
by reading node arrays is painful. Visualization accelerates all future work.

### Memory peephole optimizations

**Phase 1 — Load-Store forwarding** ✅ Done
`Load(mem, ptr)` where `mem` is `Store(_, _, ptr, value)` at same ptr
→ return `value` directly, no Load node created.

**Phase 2 — Eager dead Store elimination**
During `create_store(mem, ptr, value)`: if `mem` is `Store(mem', ptr, value')`
and the old Store has no Load users reading the same ptr, skip the
intermediate Store and chain directly off `mem'`. Catches the adjacent case.

**Phase 3 — Final dead Store sweep**
After IR generation, walk all Stores. Any Store whose memory output has
zero Loads from its ptr is dead and can be removed. Catches the tail of a
chain that Phase 2 misses.

**Phase 4 — `New` with initial value (design)**
Have `New` accept an SSA value parameter for initialization instead of
relying on zero/undef defaults. This turns Load-from-New into another
Load-Store forwarding case. Aggregate types would need an `Aggregate`
IR node to bundle field values

### End-to-end demo
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
- **Alias-class memory chains**: Replace the single `MEMORY_VAR` with
  per-class memory SSA variables. Each `New` gets tagged with an alias class;
  `Load`/`Store` only serialize within the same class. Independent operations
  on different objects become fully parallel. Same approach as JVM's C2 compiler.
  Requires: dynamic alias class allocation, per-class SSA chain, mapping from
  `New` → alias class, per-class memory Phis at control merges.