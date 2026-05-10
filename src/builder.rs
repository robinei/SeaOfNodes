use std::collections::{HashMap, HashSet};

use crate::node::*;
use crate::types::*;

/// State for a single SSA variable: its name and the reaching definition at each control point.
struct VariableState {
    name: String,
    current_def: HashMap<NodeId, NodeId>, // control node -> reaching value
}

/// Built-in variable index for the implicit memory chain.
/// Memory is tracked as an ordinary SSA variable (VarId 0), sharing the same
/// Braun-style recursive lookup, Phi insertion, and trivial Phi elimination
/// used for all other variables.
pub const MEMORY_VAR: VarId = VarId(0);

/// The main IR-building harness.
///
/// Owns the node arena, output-edge mirror, and SSA variable state.
/// Constructs nodes on-the-fly with peephole optimizations, constant folding,
/// and value numbering. Uses Braun-style lazy SSA for Phi placement.
pub struct IRBuilder {
    nodes: Vec<Node>,
    node_outputs: Vec<OutputsVec>,
    interned_nodes: HashMap<Node, NodeId>,
    current_control: NodeId,
    variables: Vec<VariableState>,
    sealed: HashSet<NodeId>,
    incomplete_phis: HashMap<(VarId, NodeId), NodeId>, // (var_idx, control) -> operandless Phi
    worklist: Vec<NodeId>,
}

// ── Test-only accessors ──

#[cfg(test)]
impl IRBuilder {
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub(crate) fn node_outputs(&self) -> &[OutputsVec] {
        &self.node_outputs
    }

    pub(crate) fn incomplete_phis(&self) -> &HashMap<(VarId, NodeId), NodeId> {
        &self.incomplete_phis
    }

    pub(crate) fn worklist(&self) -> &[NodeId] {
        &self.worklist
    }
}

impl IRBuilder {
    pub fn new() -> Self {
        // Create initial nodes: node 0 = Unreachable, node 1 = Entry, node 2 = Memory
        let mem_node = Node::new(NodeKind::Memory, Type::Memory);
        let mut builder = Self {
            nodes: vec![
                Node::new(NodeKind::Unreachable, Type::Never),
                Node::new(NodeKind::Entry, Type::Control),
                mem_node,
            ],
            node_outputs: vec![OutputsVec::new(), OutputsVec::new(), OutputsVec::new()],
            interned_nodes: HashMap::new(),
            current_control: NodeId(1), // Entry
            // Variable 0 is reserved for the implicit memory chain
            variables: vec![VariableState {
                name: "$memory".to_string(),
                current_def: HashMap::new(),
            }],
            sealed: HashSet::new(),
            incomplete_phis: HashMap::new(),
            worklist: Vec::new(),
        };
        builder.sealed.insert(NodeId(1)); // Entry is sealed from the start
        // At Entry, the reaching memory is the Memory root node (node 2)
        builder.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(NodeId(1), NodeId(2));
        builder
    }

    #[inline]
    pub fn lookup_node(&self, id: NodeId) -> &Node {
        &self.nodes[id.as_usize()]
    }

    #[inline]
    pub(crate) fn intern_node(&mut self, node: &Node) -> NodeId {
        // For pure nodes, check if already interned (avoids duplicates)
        if node.kind.is_pure() {
            if let Some(&id) = self.interned_nodes.get(node) {
                return id;
            }
        }

        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.node_outputs.push(OutputsVec::new());

        // Build outputs mirror for all inputs
        for i in 0..node.inputs_len() {
            let input_id = node.get_input(i);
            if input_id.is_valid() {
                self.node_outputs[input_id.as_usize()].push(id.0);
            }
        }

        // Insert into interned_nodes if pure
        if node.kind.is_pure() {
            self.interned_nodes.insert(node.clone(), id);
        }

        id
    }

    /// Replace an existing node entirely, keeping all outputs perfectly in sync.
    /// All users of old_id are rewired to point to the new node.
    pub fn replace_node(&mut self, old_id: NodeId, new_node: &Node) -> NodeId {
        let new_id = self.intern_node(new_node);
        self.replace_all_uses(old_id, new_id);
        new_id
    }

    /// Rewire all users of `old_id` to point to `new_id` instead.
    /// Does NOT create a new node — used by trivial Phi elimination.
    /// Also cleans up stale output edges in `old_id`'s inputs.
    pub(crate) fn replace_all_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        if old_id == new_id {
            return;
        }

        // Remove old_id from its inputs' output lists (stale edge cleanup)
        let ninputs = self.nodes[old_id.as_usize()].inputs_len();
        for i in 0..ninputs {
            let input = self.nodes[old_id.as_usize()].get_input(i);
            if input.is_valid() {
                let outputs = &mut self.node_outputs[input.as_usize()];
                if let Some(pos) = outputs.iter().position(|x| x == old_id.0) {
                    outputs.remove(pos);
                }
            }
        }

        // Snapshot old outputs, then clear the old node's output list
        let old_outputs = self.node_outputs[old_id.as_usize()].clone();
        self.node_outputs[old_id.as_usize()].set_all(&[]);

        // Rewire each old user to point to new_id instead of old_id
        // Replace ALL occurrences of old_id in each user's inputs (not just the first one).
        // A node can reference the same value multiple times, e.g., Add(x, x).
        for idx in 0..old_outputs.len() {
            let user_id = NodeId(old_outputs.get(idx));
            let kind = self.nodes[user_id.as_usize()].kind;
            let user = &mut self.nodes[user_id.as_usize()];
            for j in 0..user.inputs_len() {
                if user.get_input(j) == old_id {
                    user.set_input(j, new_id);
                }
            }
            self.node_outputs[new_id.as_usize()].push(user_id.0);

            // Schedule re-idealization for users whose input changed.
            if kind.can_idealize() {
                self.worklist.push(user_id);
            }
        }
    }

    /// Allocate a new source variable with a human-readable name.
    /// Returns a VarId starting from 1 (VarId(0) is reserved for the memory chain).
    pub fn create_variable(&mut self, name: &str) -> VarId {
        let idx = self.variables.len();
        self.variables.push(VariableState {
            name: name.to_string(),
            current_def: HashMap::new(),
        });
        VarId(idx)
    }

    /// Set the current control (program point) for subsequent reads/writes.
    #[inline]
    pub fn set_control(&mut self, control: NodeId) {
        self.current_control = control;
    }

    /// Get the current control node.
    #[inline]
    pub fn current_control(&self) -> NodeId {
        self.current_control
    }

    /// Record that `value` is the current definition of `var` at the current control point.
    #[inline]
    pub fn write_variable(&mut self, var: VarId, value: NodeId) {
        self.variables[var.as_usize()]
            .current_def
            .insert(self.current_control, value);
    }

    /// Read the current value of `var` at the current control point.
    /// Inserts Phi nodes as needed (Braun-style lazy SSA construction).
    pub fn read_variable(&mut self, var: VarId) -> NodeId {
        self.lookup_variable(var, self.current_control)
    }

    /// Internal: look up a variable at a specific control point, recursing backward if needed.
    fn lookup_variable(&mut self, var: VarId, ctrl: NodeId) -> NodeId {
        // Local value numbering check
        if let Some(&val) = self.variables[var.as_usize()].current_def.get(&ctrl) {
            return val;
        }
        self.read_variable_recursive(var, ctrl)
    }

    /// Braun-style backward search for a variable's reaching definition.
    /// TODO: Consider adding a recursion depth limit as a safety net — the
    /// Braun backward search can recurse arbitrarily deep through predecessor
    /// chains (terminates in practice but no explicit safeguard).
    fn read_variable_recursive(&mut self, var: VarId, ctrl: NodeId) -> NodeId {
        if !self.sealed.contains(&ctrl) {
            // Incomplete CFG: place an operandless Phi and record it as incomplete
            let phi = self.create_phi_node(ctrl, &[]);
            let phi_id = self.intern_node(&phi);
            self.incomplete_phis.insert((var, ctrl), phi_id);
            self.variables[var.as_usize()].current_def.insert(ctrl, phi_id);
            return phi_id;
        }

        let preds = self.get_control_predecessors(ctrl);
        if preds.is_empty() {
            // No predecessors (e.g., Entry): variable is undefined → NodeId 0 (Unreachable)
            self.variables[var.as_usize()].current_def.insert(ctrl, NodeId(0));
            return NodeId(0);
        }
        if preds.len() == 1 {
            // Single predecessor: just recurse, no Phi needed
            let val = self.lookup_variable(var, NodeId(preds.get(0)));
            self.variables[var.as_usize()].current_def.insert(ctrl, val);
            return val;
        }

        // Multiple predecessors: place operandless Phi to break cycles, then fill operands
        let phi = self.create_phi_node(ctrl, &[]);
        let phi_id = self.intern_node(&phi);
        self.variables[var.as_usize()].current_def.insert(ctrl, phi_id);
        let val = self.add_phi_operands(var, phi_id, ctrl);
        self.variables[var.as_usize()].current_def.insert(ctrl, val);
        val
    }

    /// Fill operands of a newly-created Phi by recursively reading from each predecessor.
    /// Then attempt trivial Phi elimination.
    /// TODO: Consider deduplicating identical operands from different predecessors
    /// to avoid bloated Phis that immediately collapse via try_remove_trivial_phi.
    fn add_phi_operands(&mut self, var: VarId, phi_id: NodeId, ctrl: NodeId) -> NodeId {
        let preds = self.get_control_predecessors(ctrl);
        let mut operand_types: Vec<Type> = Vec::new();
        for raw in preds.iter() {
            let pred = NodeId(raw);
            let val = self.lookup_variable(var, pred);
            // Append operand to Phi
            self.nodes[phi_id.as_usize()].push_input(val);
            // Register phi_id as a user of val
            self.node_outputs[val.as_usize()].push(phi_id.0);
            // Collect operand type for refining Phi's type
            operand_types.push(self.nodes[val.as_usize()].t.clone());
        }
        // Refine Phi's type: join of all operand types
        // For memory Phis, the type is always Memory regardless of operand types
        // (memory operands are tokens in a chain, their node types are irrelevant)
        if var == MEMORY_VAR {
            self.nodes[phi_id.as_usize()].t = Type::Memory;
        } else if operand_types.is_empty() {
            self.nodes[phi_id.as_usize()].t = Type::Never;
        } else if operand_types.iter().all(|t| *t == operand_types[0]) {
            self.nodes[phi_id.as_usize()].t = operand_types[0].clone();
        } else {
            self.nodes[phi_id.as_usize()].t = Type::make_union(operand_types);
        }
        self.try_remove_trivial_phi(phi_id)
    }

    /// Braun Algorithm 3: detect and remove a trivial Phi.
    /// A Phi is trivial if it merges the same value (possibly with self-references).
    /// Called eagerly during SSA construction (from add_phi_operands).
    /// The cascading check of downstream Phis is deferred to the worklist.
    fn try_remove_trivial_phi(&mut self, phi_id: NodeId) -> NodeId {
        let replacement = match self.is_trivial_phi(phi_id) {
            Some(v) => v,
            None => return phi_id,
        };

        // Reroute all uses of phi to replacement.
        // replace_all_uses pushes affected Phi users to the worklist, so
        // cascading trivial-Phi detection happens during worklist drain.
        self.replace_all_uses(phi_id, replacement);
        replacement
    }

    /// Check if a Phi is trivial — all value operands agree (ignoring self-refs).
    /// Returns `Some(replacement)` if trivial, `None` otherwise.
    fn is_trivial_phi(&self, phi_id: NodeId) -> Option<NodeId> {
        let mut same: Option<NodeId> = None;
        let phi = &self.nodes[phi_id.as_usize()];

        // Skip input[0] (the control region); only consider value operands
        for i in 1..phi.inputs_len() {
            let op = phi.get_input(i);
            if op == phi_id || Some(op) == same {
                continue;
            }
            if same.is_some() {
                return None; // At least two distinct values: not trivial
            }
            same = Some(op);
        }

        Some(match same {
            Some(v) => v,
            None => NodeId(0), // Phi references only itself (unreachable) → undef
        })
    }

    // ── Memory Convenience Methods ──

    /// Read the current memory at the current control point.
    /// Inserts memory Phis as needed (same Braun algorithm as variables).
    #[inline]
    pub fn get_current_memory(&mut self) -> NodeId {
        self.read_variable(MEMORY_VAR)
    }

    /// Set the current memory at the current control point.
    #[inline]
    pub fn set_current_memory(&mut self, mem: NodeId) {
        self.write_variable(MEMORY_VAR, mem);
    }

    /// Get the control predecessors of a control node.
    fn get_control_predecessors(&self, ctrl: NodeId) -> OutputsVec {
        let mut preds = OutputsVec::new();
        match self.nodes[ctrl.as_usize()].kind {
            NodeKind::Region | NodeKind::Loop => {
                // All inputs are control predecessors
                let n = self.nodes[ctrl.as_usize()].inputs_len();
                for i in 0..n {
                    preds.push(self.nodes[ctrl.as_usize()].get_input(i).0);
                }
            }
            NodeKind::Entry => {}
            _ => {
                // For non-merge control nodes, the single control input at index 0
                if self.nodes[ctrl.as_usize()].inputs_len() > 0 {
                    preds.push(self.nodes[ctrl.as_usize()].get_input(0).0);
                }
            }
        }
        preds
    }

    /// Add a new predecessor to a Loop node (for back-edges).
    /// The loop is not sealed yet, so no Phi filling is triggered.
    /// TODO: Add guard against adding back-edges after sealing — the protocol
    /// requires all back-edges be added before seal() is called. If a back-edge
    /// is added after sealing, Phis placed earlier would miss that predecessor.
    pub fn push_loop_back_edge(&mut self, loop_ctrl: NodeId, back_edge: NodeId) {
        self.nodes[loop_ctrl.as_usize()].push_input(back_edge);
        // Register the back-edge as a user of loop_ctrl's outputs...
        // Actually, back_edge is an input to the loop node, so loop_ctrl
        // becomes an output of back_edge.
        self.node_outputs[back_edge.as_usize()].push(loop_ctrl.0);
    }

    /// Seal a control node — no more predecessors will be added.
    /// Fills in any incomplete Phis (including memory Phis, since memory is VarId 0)
    /// that were placed before sealing.
    /// TODO: Consider verifying that all back-edges have been added before sealing
    /// a Loop node. Currently the protocol is caller-enforced (add back-edges, then seal).
    pub fn seal(&mut self, ctrl: NodeId) {
        self.sealed.insert(ctrl);

        // Collect incomplete Phis for this block and fill their operands
        let keys_to_fill: Vec<(VarId, NodeId)> = self
            .incomplete_phis
            .keys()
            .filter(|&&(_, c)| c == ctrl)
            .cloned()
            .collect();

        for key in keys_to_fill {
            let phi_id = self.incomplete_phis.remove(&key).unwrap();
            let (var, _) = key;
            let result = self.add_phi_operands(var, phi_id, ctrl);
            self.variables[var.as_usize()].current_def.insert(ctrl, result);
        }
    }

    /// Create a Phi node with the given control and value inputs.
    /// The type starts as Never and is refined in add_phi_operands.
    /// TODO: The intermediate Never type between creation and operand filling
    /// could confuse debugging/inspection. Consider using a dedicated placeholder.
    fn create_phi_node(&self, ctrl: NodeId, values: &[NodeId]) -> Node {
        let mut node = Node::new(NodeKind::Phi, Type::Never);
        let mut inputs = vec![ctrl];
        inputs.extend_from_slice(values);
        node.set_inputs(&inputs);
        node
    }

    #[inline]
    pub(crate) fn get_node_id(&mut self, node: &Node) -> NodeId {
        if let Some(identity_id) = node.get_identity_id() {
            identity_id
        } else if node.kind.is_pure() {
            self.intern_node(node) // intern_node handles interning
        } else {
            panic!("can't intern impure node - they need to have identity")
        }
    }

    #[inline]
    fn resolve_if_identity_node<'a>(&'a self, node: &'a Node) -> &'a Node {
        if let Some(identity_id) = node.get_identity_id() {
            &self.nodes[identity_id.as_usize()]
        } else {
            node
        }
    }

    // ── Arithmetic Builder Methods ──

    pub fn create_add(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.add(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            // 0 + x => x
            rhs.clone()
        } else if r.t.is_const_zero() {
            // x + 0 => x
            lhs.clone()
        } else if l == r {
            // x + x => x * 2
            self.create_mul(lhs, &Node::num_of_type(2, &t))?
        } else if l.kind != NodeKind::Add && r.kind == NodeKind::Add {
            // non-add + add => add + non-add
            self.create_add(rhs, lhs)?
        } else if r.kind == NodeKind::Neg {
            // x + (-y) => x - y
            let y = self.lookup_node(r.value()).clone();
            self.create_sub(lhs, &y)?
        } else if r.kind == NodeKind::Add {
            // x + (y + z) => (x + y) + z
            let y = self.lookup_node(r.lhs()).clone();
            let z = self.lookup_node(r.rhs()).clone();
            let xy = self.create_add(lhs, &y)?;
            self.create_add(&xy, &z)?
        } else {
            let mut node = Node::new(NodeKind::Add, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        })
    }

    pub fn create_sub(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.sub(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if r.t.is_const_zero() {
            // x - 0 => x
            lhs.clone()
        } else if l == r {
            // x - x => 0
            Node::num_of_type(0, &t)
        } else if r.kind == NodeKind::Neg {
            // x - (-y) => x + y
            let y = self.lookup_node(r.value()).clone();
            self.create_add(lhs, &y)?
        } else if r.kind == NodeKind::Sub {
            // x - (y - z) => x - y + z => (x + z) - y
            let y = self.lookup_node(r.lhs()).clone();
            let z = self.lookup_node(r.rhs()).clone();
            let xz = self.create_add(lhs, &z)?;
            self.create_sub(&xz, &y)?
        } else {
            let mut node = Node::new(NodeKind::Sub, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        })
    }

    pub fn create_mul(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.mul(&r.t)?;

        Ok(if t.is_const() {
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() || r.t.is_const_zero() {
            Node::num_of_type(0, &t)
        } else if l.t.is_const_one() {
            rhs.clone()
        } else if r.t.is_const_one() {
            lhs.clone()
        } else if l == r {
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        } else if l.kind != NodeKind::Mul && r.kind == NodeKind::Mul {
            self.create_mul(rhs, lhs)?
        } else {
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        })
    }

    pub fn create_div(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.div(&r.t)?;

        Ok(if t.is_const() {
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            Node::num_of_type(0, &t)
        } else if r.t.is_const_one() {
            lhs.clone()
        } else if l == r {
            Node::num_of_type(1, &t)
        } else {
            let mut node = Node::new(NodeKind::Div, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        })
    }

    // ── Control Flow Builder Methods ──

    pub fn create_if(&mut self, control: NodeId, cond: &Node) -> NodeId {
        let mut node = Node::new(NodeKind::If, Type::Control);
        node.set_inputs(&[control, self.get_node_id(cond)]);
        let id = self.intern_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_then(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Then, Type::Control);
        node.set_inputs(&[if_node]);
        let id = self.intern_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_else(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Else, Type::Control);
        node.set_inputs(&[if_node]);
        let id = self.intern_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_loop(&mut self, control: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Loop, Type::Control);
        node.set_inputs(&[control]);
        self.intern_node(&node)
    }

    pub fn create_region(&mut self, controls: &[NodeId]) -> NodeId {
        let mut node = Node::new(NodeKind::Region, Type::Control);
        node.set_inputs(controls);
        let id = self.intern_node(&node);
        self.seal(id);
        id
    }

    // ── Param / Cast Builder Methods ──

    pub fn create_param(&mut self, index: usize, t: Type) -> Node {
        let mut param_node = Node::new(NodeKind::Param, t.clone());
        param_node.data.param_index = index;
        let param_id = self.intern_node(&param_node);

        let mut identity = Node::new(NodeKind::Identity, t);
        identity.data.identity_id = param_id;
        identity
    }

    pub fn create_cast(&mut self, value: &Node, target_type: Type) -> Result<Node, IRError> {
        use crate::types::CastKind;

        match value.t.cast_kind(&target_type) {
            CastKind::Static => {
                if value.t == target_type {
                    Ok(value.clone())
                } else {
                    let mut node = Node::new(NodeKind::StaticCast, target_type);
                    node.set_inputs(&[NodeId(0), self.get_node_id(value)]);
                    Ok(node)
                }
            }
            CastKind::Dynamic => {
                let mut node = Node::new(NodeKind::DynamicCast, target_type);
                node.set_inputs(&[NodeId(0), self.get_node_id(value)]);
                Ok(node)
            }
            CastKind::Invalid => Err(IRError::TypeMismatch),
        }
    }

    // ── Memory Builder Methods ──

    /// Create a New (allocation) node.
    pub fn create_new(&mut self, mem: NodeId, alloc_type: Type) -> NodeId {
        let control = self.current_control;
        let mut node = Node::new(NodeKind::New, alloc_type);
        node.set_inputs(&[control, mem]);
        let id = self.intern_node(&node);
        self.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(control, id);
        id
    }

    /// Create a Load (memory read) node with on-the-fly Load-Store forwarding.
    /// TODO: Load-Store forwarding only checks the immediate memory predecessor.
    /// Consider walking back through intermediate Stores to different ptrs:
    ///   Store(ptr, 10) -> Store(ptr1, 7) -> Load(ptr, mem=Store(ptr1,7))
    /// currently misses forwarding the value 10 from the earlier Store.
    pub fn create_load(&mut self, mem: NodeId, ptr: NodeId, loaded_type: Type) -> NodeId {
        // Peephole: Load-Store forwarding
        let mem_node = &self.nodes[mem.as_usize()];
        if mem_node.kind == NodeKind::Store && mem_node.get_input(2) == ptr {
            return mem_node.get_input(3);
        }

        let control = self.current_control;
        let mut node = Node::new(NodeKind::Load, loaded_type);
        node.set_inputs(&[control, mem, ptr]);
        self.intern_node(&node)
    }

    /// Create a Store (memory write) node.
    pub fn create_store(&mut self, mem: NodeId, ptr: NodeId, value: NodeId) -> NodeId {
        let control = self.current_control;
        let mut node = Node::new(NodeKind::Store, Type::Memory);
        node.set_inputs(&[control, mem, ptr, value]);
        let id = self.intern_node(&node);
        self.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(control, id);
        id
    }

    // ── Worklist / Idealization ──

    /// Wrap a node ID in an Identity facade when needed for the creator API.
    /// Builders expect &Node references (Potentially Identity-wrapped for Param-like nodes).
    /// When reading a node from the arena that is non-pure and non-Identity,
    /// wrap it in an Identity so that the creator's resolve_if_identity_node works correctly
    /// and the creator's fallback get_node_id call doesn't panic.
    fn wrap_for_creator(&self, id: NodeId) -> Node {
        let node = self.lookup_node(id);
        if node.kind.is_pure() || node.kind == NodeKind::Identity {
            node.clone()
        } else {
            let mut identity = Node::new(NodeKind::Identity, node.t.clone());
            identity.data.identity_id = id;
            identity
        }
    }

    /// Re-idealize a single node slot by re-running its creator with current inputs.
    /// Returns `Some(new_id)` if the node was simplified to a different existing node,
    /// or `None` if it's already ideal.
    fn idealize(&mut self, slot: NodeId) -> Option<NodeId> {
        let kind = self.nodes[slot.as_usize()].kind;

        // Phis are handled with a simple structural check (triviality)
        // rather than re-running a creator function.
        if kind == NodeKind::Phi {
            return self.is_trivial_phi(slot).filter(|&id| id != slot);
        }

        let (lhs, rhs): (NodeId, NodeId) = {
            let node = &self.nodes[slot.as_usize()];
            let lhs = if node.inputs_len() > 1 { node.get_input(1) } else { NodeId(0) };
            let rhs = if node.inputs_len() > 2 { node.get_input(2) } else { NodeId(0) };
            (lhs, rhs)
        };

        let result = match kind {
            NodeKind::Add => {
                let lhs = self.wrap_for_creator(lhs);
                let rhs = self.wrap_for_creator(rhs);
                self.create_add(&lhs, &rhs).ok()
            }
            NodeKind::Sub => {
                let lhs = self.wrap_for_creator(lhs);
                let rhs = self.wrap_for_creator(rhs);
                self.create_sub(&lhs, &rhs).ok()
            }
            NodeKind::Mul => {
                let lhs = self.wrap_for_creator(lhs);
                let rhs = self.wrap_for_creator(rhs);
                self.create_mul(&lhs, &rhs).ok()
            }
            NodeKind::Div => {
                let lhs = self.wrap_for_creator(lhs);
                let rhs = self.wrap_for_creator(rhs);
                self.create_div(&lhs, &rhs).ok()
            }
            _ => None,
        };

        match result {
            Some(node) => {
                let new_id = match node.get_identity_id() {
                    Some(id) => id,
                    None => self.intern_node(&node),
                };
                if new_id != slot {
                    Some(new_id)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Drain the worklist, re-idealizing any nodes whose inputs changed.
    /// This ensures cascading optimizations propagate (e.g., a Phi collapse
    /// triggers re-evaluation of its pure-node users).
    pub fn process_worklist(&mut self) {
        while let Some(slot) = self.worklist.pop() {
            if let Some(new_id) = self.idealize(slot) {
                self.replace_all_uses(slot, new_id);
            }
        }
    }

    /// Returns true if there are pending work items to process.
    pub fn has_work(&self) -> bool {
        !self.worklist.is_empty()
    }

    /// Serialize the graph to Graphviz DOT format for debugging.
    /// Variable names (from SSA variable tracking) are included in node labels.
    pub fn to_dot(&self) -> String {
        let mut var_labels = std::collections::HashMap::new();
        for var in &self.variables {
            for (_, &node_id) in &var.current_def {
                let entry = var_labels.entry(node_id).or_insert_with(String::new);
                if !entry.is_empty() {
                    entry.push_str(", ");
                }
                entry.push_str(&var.name);
            }
        }
        crate::dot::graph_to_dot(&self.nodes, &self.node_outputs, &var_labels)
    }
}