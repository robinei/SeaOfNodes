mod compact_vec;
mod constraints;
mod symbols;
mod types;
use compact_vec::{CompactVecData, CompactVecState, CompactVecStateU16};
use std::collections::{HashMap, HashSet};
use types::*;

use crate::compact_vec::CompactVec;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const UNREACHABLE: NodeId = NodeId(0);

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != 0
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(v: u32) -> Self {
        NodeId(v)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NodeId> for u32 {
    #[inline]
    fn from(id: NodeId) -> Self {
        id.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VarId(pub usize);

impl VarId {
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl From<usize> for VarId {
    #[inline]
    fn from(v: usize) -> Self {
        VarId(v)
    }
}

type InputsVecData = CompactVecData<16, 8, 4>;
type OutputsVec = CompactVec<16, 8, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IRError {
    TypeMismatch,
    IntegerOverflow,
    DivisionByZero,
    InvalidPrimitiveCoercion,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Unreachable,

    // fake node which contains the ID of node which is stored in the nodes array (in data.identity_id).
    // lets us use the Node value based interface even for nodes with identity
    // inputs: none
    Identity,

    // function parameter (just an index stored in data.param_index).
    // inputs: none
    Param,

    // constant value (value stored in data field, type will be one of Const*)
    // inputs: none
    Const,

    // data merge node
    // inputs: control (region), value per predecessor
    Phi,

    // control nodes
    Entry,  // entry control node (aka start). inputs: none
    If,     // inputs: control, predicate
    Then,   // then branch projection. inputs: control (if)
    Else,   // else brench projection. inputs: control (if)
    Region, // control merge node. inputs: control per predecessor
    Loop,   // special control merge node for loops. inputs: control per predecessor

    // memory nodes
    Memory, // initial mem state. root of all mem chains. value is mem. inputs: none
    New,    // allocate new object of type matching node type. value is ptr. inputs: mem
    Load, // load from ptr. value is loaded data, but can polymorphically act as mem. inputs: mem, ptr (from new), offset
    Store, // store to ptr. value is mem. inputs: mem,  ptr (from new), value

    // unary operations
    // inputs: value
    Neg,
    Not,

    // binary operations
    // inputs: lhs, rhs
    Add,
    Sub,
    Mul,
    Div,

    // cast operations
    // inputs: control, value
    StaticCast,  // safe cast, no runtime check needed
    DynamicCast, // cast requiring runtime type check
}

impl NodeKind {
    pub fn is_pure(self) -> bool {
        use NodeKind::*;
        matches!(
            self,
            Const | Neg | Not | Add | Sub | Mul | Div | StaticCast | DynamicCast
        )
    }
}

#[derive(Copy, Clone)]
union NodeData {
    // Compact vector for inputs
    inputs: InputsVecData,

    // Other node data
    param_index: usize,
    identity_id: NodeId,
}

impl std::fmt::Debug for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Since this is a union, we can't safely access all fields
        // Just show it as opaque data
        f.debug_struct("NodeData").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    kind: NodeKind,
    _flags: u8,
    inputs_state: CompactVecStateU16,
    _pad: u32,
    t: Type,
    data: NodeData,
}

impl Node {
    #[inline]
    pub fn new(kind: NodeKind, t: Type) -> Self {
        let mut node = Self::default();
        node.kind = kind;
        node.t = t;
        node.inputs_state = CompactVecStateU16::new_empty();
        node
    }

    #[inline]
    pub fn num_of_type(value: u32, t: &Type) -> Node {
        match t {
            Type::Int(..) => Node::const_int(value as i64),
            Type::UInt(..) => Node::const_uint(value as u64),
            Type::Float(..) => Node::const_float(value as f64),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn const_bool(value: bool) -> Self {
        Self::new(NodeKind::Const, Type::const_bool(value))
    }

    #[inline]
    pub fn const_int(value: i64) -> Self {
        Self::new(NodeKind::Const, Type::const_int(value))
    }

    #[inline]
    pub fn const_uint(value: u64) -> Self {
        Self::new(NodeKind::Const, Type::const_uint(value))
    }

    #[inline]
    pub fn const_float(value: f64) -> Self {
        Self::new(NodeKind::Const, Type::const_float(value))
    }

    #[inline]
    pub fn inputs_len(&self) -> usize {
        self.inputs_state.len()
    }

    #[inline]
    pub fn get_input(&self, index: usize) -> NodeId {
        NodeId(unsafe { self.data.inputs.get(&self.inputs_state, index) })
    }

    #[inline]
    pub fn set_inputs(&mut self, inputs: &[NodeId]) {
        let raw: Vec<u32> = inputs.iter().map(|id| id.0).collect();
        unsafe {
            self.data.inputs.set_all(&mut self.inputs_state, &raw);
        }
    }

    #[inline]
    pub fn set_input(&mut self, index: usize, input: NodeId) {
        unsafe {
            self.data.inputs.set(&mut self.inputs_state, index, input.0);
        }
    }

    #[inline]
    pub fn push_input(&mut self, input: NodeId) {
        unsafe {
            self.data.inputs.push(&mut self.inputs_state, input.0);
        }
    }

    #[inline]
    pub fn get_identity_id(&self) -> Option<NodeId> {
        if self.kind == NodeKind::Identity {
            Some(unsafe { self.data.identity_id })
        } else {
            None
        }
    }
}

impl Default for Node {
    #[inline]
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

impl std::hash::Hash for Node {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.t.hash(state);
        self.inputs_state.hash(state);
        for i in 0..self.inputs_len() {
            self.get_input(i).hash(state);
        }
    }
}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.kind != other.kind
            || self.t != other.t
            || self.inputs_state != other.inputs_state
        {
            return false;
        }
        let len = self.inputs_len();
        if len != other.inputs_len() {
            return false;
        }
        for i in 0..len {
            if self.get_input(i) != other.get_input(i) {
                return false;
            }
        }
        true
    }
}

impl Eq for Node {}

impl Drop for Node {
    #[inline]
    fn drop(&mut self) {
        // Drop heap-allocated input storage if present
        unsafe {
            self.data.inputs.drop_heap(&self.inputs_state);
        }
    }
}

struct VariableState {
    name: String,
    current_def: HashMap<NodeId, NodeId>, // control node -> reaching value
}

/// Built-in variable index for the implicit memory chain.
/// Memory is tracked as an ordinary SSA variable (VarId 0), sharing the same
/// Braun-style recursive lookup, Phi insertion, and trivial Phi elimination
/// used for all other variables.
pub const MEMORY_VAR: VarId = VarId(0);

pub struct IRBuilder {
    nodes: Vec<Node>,
    node_outputs: Vec<OutputsVec>,
    interned_nodes: HashMap<Node, NodeId>,
    current_control: NodeId,
    variables: Vec<VariableState>,
    sealed: HashSet<NodeId>,
    incomplete_phis: HashMap<(VarId, NodeId), NodeId>, // (var_idx, control) -> operandless Phi
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
        };
        builder.sealed.insert(NodeId(1)); // Entry is sealed from the start
        // At Entry, the reaching memory is the Memory root node (node 2)
        builder.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(NodeId(1), NodeId(2));
        builder
    }

    #[inline]
    fn lookup_node(&self, id: NodeId) -> &Node {
        &self.nodes[id.as_usize()]
    }

    #[inline]
    fn push_node(&mut self, node: &Node) -> NodeId {
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
        let new_id = self.push_node(new_node);
        self.replace_all_uses(old_id, new_id);
        new_id
    }

    /// Rewire all users of `old_id` to point to `new_id` instead.
    /// Does NOT create a new node — used by trivial Phi elimination.
    /// Also cleans up stale output edges in `old_id`'s inputs.
    fn replace_all_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        if old_id == new_id {
            return;
        }

        // Remove old_id from its inputs' output lists (stale edge cleanup)
        let inputs: Vec<NodeId> = {
            let old_node = &self.nodes[old_id.as_usize()];
            (0..old_node.inputs_len()).map(|i| old_node.get_input(i)).collect()
        };
        for &input in &inputs {
            if input.is_valid() {
                let outputs = &mut self.node_outputs[input.as_usize()];
                if let Some(pos) = outputs.iter().position(|x| x == old_id.0) {
                    outputs.remove(pos);
                }
            }
        }

        // Snapshot old outputs, then clear the old node's output list
        let old_outputs: Vec<NodeId> = self.node_outputs[old_id.as_usize()]
            .iter()
            .map(|x| NodeId(x))
            .collect();
        self.node_outputs[old_id.as_usize()].set_all(&[]);

        // Rewire each old user to point to new_id instead of old_id
        for &user_id in &old_outputs {
            let user = &mut self.nodes[user_id.as_usize()];
            for i in 0..user.inputs_len() {
                if user.get_input(i) == old_id {
                    user.set_input(i, new_id);
                    break;
                }
            }
            self.node_outputs[new_id.as_usize()].push(user_id.0);
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
    fn read_variable_recursive(&mut self, var: VarId, ctrl: NodeId) -> NodeId {
        if !self.sealed.contains(&ctrl) {
            // Incomplete CFG: place an operandless Phi and record it as incomplete
            let phi = self.create_phi_node(ctrl, &[]);
            let phi_id = self.push_node(&phi);
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
            let val = self.lookup_variable(var, preds[0]);
            self.variables[var.as_usize()].current_def.insert(ctrl, val);
            return val;
        }

        // Multiple predecessors: place operandless Phi to break cycles, then fill operands
        let phi = self.create_phi_node(ctrl, &[]);
        let phi_id = self.push_node(&phi);
        self.variables[var.as_usize()].current_def.insert(ctrl, phi_id);
        let val = self.add_phi_operands(var, phi_id, ctrl);
        self.variables[var.as_usize()].current_def.insert(ctrl, val);
        val
    }

    /// Fill operands of a newly-created Phi by recursively reading from each predecessor.
    /// Then attempt trivial Phi elimination.
    fn add_phi_operands(&mut self, var: VarId, phi_id: NodeId, ctrl: NodeId) -> NodeId {
        let preds = self.get_control_predecessors(ctrl);
        let mut operand_types: Vec<Type> = Vec::new();
        for &pred in &preds {
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
    fn try_remove_trivial_phi(&mut self, phi_id: NodeId) -> NodeId {
        let mut same: Option<NodeId> = None;
        let phi = &self.nodes[phi_id.as_usize()];

        // Skip input[0] (the control region); only consider value operands
        for i in 1..phi.inputs_len() {
            let op = phi.get_input(i);
            if op == phi_id || Some(op) == same {
                continue;
            }
            if same.is_some() {
                return phi_id; // At least two distinct values: not trivial
            }
            same = Some(op);
        }

        let replacement = match same {
            Some(v) => v,
            None => NodeId(0), // Phi references only itself (unreachable) → undef = node 0 (Unreachable)
        };

        // Snapshot phi's users before rerouting
        let phi_users: Vec<NodeId> = self.node_outputs[phi_id.as_usize()].iter().map(|x| NodeId(x)).collect();

        // Reroute all uses of phi to replacement
        self.replace_all_uses(phi_id, replacement);

        // Recursively check Phi users that might have become trivial
        for &user_id in &phi_users {
            if user_id != phi_id {
                let kind = self.nodes[user_id.as_usize()].kind;
                if kind == NodeKind::Phi {
                    self.try_remove_trivial_phi(user_id);
                }
            }
        }

        replacement
    }

        // ── Memory Convenience Methods ──
    // Memory is tracked as an ordinary SSA variable (MEMORY_VAR = VarId(0)).
    // These are thin wrappers for readability.

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
    fn get_control_predecessors(&self, ctrl: NodeId) -> Vec<NodeId> {
        match self.nodes[ctrl.as_usize()].kind {
            NodeKind::Region | NodeKind::Loop => {
                // All inputs are control predecessors
                let n = self.nodes[ctrl.as_usize()].inputs_len();
                (0..n).map(|i| self.nodes[ctrl.as_usize()].get_input(i)).collect()
            }
            NodeKind::Entry => vec![],
            _ => {
                // For non-merge control nodes, the single control input at index 0
                if self.nodes[ctrl.as_usize()].inputs_len() > 0 {
                    vec![self.nodes[ctrl.as_usize()].get_input(0)]
                } else {
                    vec![]
                }
            }
        }
    }

    /// Add a new predecessor to a Loop node (for back-edges).
    /// The loop is not sealed yet, so no Phi filling is triggered.
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
    fn create_phi_node(&self, ctrl: NodeId, values: &[NodeId]) -> Node {
        // The type will be refined as operands are added, for now use Never
        let mut node = Node::new(NodeKind::Phi, Type::Never);
        let mut inputs = vec![ctrl];
        inputs.extend_from_slice(values);
        node.set_inputs(&inputs);
        node
    }

    #[inline]
    fn get_node_id(&mut self, node: &Node) -> NodeId {
        if let Some(identity_id) = node.get_identity_id() {
            identity_id
        } else if node.kind.is_pure() {
            self.push_node(node) // push_node handles interning
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
            let y = self.lookup_node(r.get_input(1)).clone(); // Neg has input at index 1
            self.create_sub(lhs, &y)?
        } else if r.kind == NodeKind::Add {
            // x + (y + z) => (x + y) + z
            let y = self.lookup_node(r.get_input(1)).clone(); // Left operand at index 1
            let z = self.lookup_node(r.get_input(2)).clone(); // Right operand at index 2
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
            let y = self.lookup_node(r.get_input(1)).clone();
            self.create_add(lhs, &y)?
        } else if r.kind == NodeKind::Sub {
            // x - (y - z) => x - y + z => (x + z) - y
            let y = self.lookup_node(r.get_input(1)).clone(); // Left operand at index 1
            let z = self.lookup_node(r.get_input(2)).clone(); // Right operand at index 2
            let xz = self.create_add(lhs, &z)?;
            self.create_sub(&xz, &y)?
        } else {
            let mut node = Node::new(NodeKind::Sub, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_mul(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.mul(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() || r.t.is_const_zero() {
            // x * 0 => 0 or 0 * x => 0
            Node::num_of_type(0, &t)
        } else if l.t.is_const_one() {
            // 1 * x => x
            rhs.clone()
        } else if r.t.is_const_one() {
            // x * 1 => x
            lhs.clone()
        } else if l == r {
            // x * x => x^2 (no special peephole for now)
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        } else if l.kind != NodeKind::Mul && r.kind == NodeKind::Mul {
            // non-mul * mul => mul * non-mul (canonicalize)
            self.create_mul(rhs, lhs)?
        } else {
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_div(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve_if_identity_node(lhs);
        let r = self.resolve_if_identity_node(rhs);

        let t = l.t.div(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            // 0 / x => 0 (assuming x != 0, which is checked by t.div())
            Node::num_of_type(0, &t)
        } else if r.t.is_const_one() {
            // x / 1 => x
            lhs.clone()
        } else if l == r {
            // x / x => 1 (assuming x != 0, which is checked by t.div())
            Node::num_of_type(1, &t)
        } else {
            let mut node = Node::new(NodeKind::Div, t);
            node.set_inputs(&[NodeId(0), self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_if(&mut self, control: NodeId, cond: &Node) -> NodeId {
        let mut node = Node::new(NodeKind::If, Type::Control);
        node.set_inputs(&[control, self.get_node_id(cond)]);
        let id = self.push_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_then(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Then, Type::Control);
        node.set_inputs(&[if_node]);
        let id = self.push_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_else(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Else, Type::Control);
        node.set_inputs(&[if_node]);
        let id = self.push_node(&node);
        self.sealed.insert(id);
        id
    }

    pub fn create_loop(&mut self, control: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Loop, Type::Control);
        node.set_inputs(&[control]);
        self.push_node(&node)
    }

    pub fn create_region(&mut self, controls: &[NodeId]) -> NodeId {
        let mut node = Node::new(NodeKind::Region, Type::Control);
        node.set_inputs(controls);
        let id = self.push_node(&node);
        self.seal(id);
        id
    }

    pub fn create_param(&mut self, index: usize, t: Type) -> Node {
        let mut param_node = Node::new(NodeKind::Param, t.clone());
        param_node.data.param_index = index;
        let param_id = self.push_node(&param_node);

        let mut identity = Node::new(NodeKind::Identity, t);
        identity.data.identity_id = param_id;
        identity
    }

    pub fn create_cast(&mut self, value: &Node, target_type: Type) -> Result<Node, IRError> {
        use crate::types::CastKind;

        match value.t.cast_kind(&target_type) {
            CastKind::Static => {
                // Identity cast - just return the original node with new type
                if value.t == target_type {
                    Ok(value.clone())
                } else {
                    // Safe cast, create StaticCast node
                    let mut node = Node::new(NodeKind::StaticCast, target_type);
                    node.set_inputs(&[NodeId(0), self.get_node_id(value)]);
                    Ok(node)
                }
            }

            CastKind::Dynamic => {
                // Runtime check required, create DynamicCast node
                let mut node = Node::new(NodeKind::DynamicCast, target_type);
                node.set_inputs(&[NodeId(0), self.get_node_id(value)]);
                Ok(node)
            }

            CastKind::Invalid => {
                // Cast is impossible
                Err(IRError::TypeMismatch)
            }
        }
    }

    // ── Memory Builder Methods ──

    /// Create a New (allocation) node.
    /// `mem` is the current memory state (before allocation).
    /// `alloc_type` is the type of the allocated object (the pointer type).
    /// Returns a Node whose type is `alloc_type` (the pointer) and whose NodeId
    /// represents the new memory state after allocation.
    /// Automatically updates the current memory at the current control point.
    pub fn create_new(&mut self, mem: NodeId, alloc_type: Type) -> NodeId {
        let control = self.current_control;
        let mut node = Node::new(NodeKind::New, alloc_type);
        node.set_inputs(&[control, mem]);
        let id = self.push_node(&node);
        // New produces a new memory state — update the memory variable
        self.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(control, id);
        id
    }

    /// Create a Load (memory read) node.
    /// `mem` is the memory state to read from.
    /// `ptr` is the pointer (from a New node) to read from.
    /// `loaded_type` is the type of the loaded value.
    /// Returns a Node whose type is `loaded_type`.
    /// Load is "amphibious" — it doesn't produce a new memory state, just reads.
    ///
    /// On-the-fly peephole — Load-Store forwarding:
    /// If `mem` is a `Store` to the same `ptr`, return the Store's value directly
    /// instead of creating a Load node.
    pub fn create_load(&mut self, mem: NodeId, ptr: NodeId, loaded_type: Type) -> NodeId {
        // Peephole: Load-Store forwarding
        // Load(mem, ptr) where mem is Store(_, _, ptr, value) → value
        let mem_node = &self.nodes[mem.as_usize()];
        if mem_node.kind == NodeKind::Store && mem_node.get_input(2) == ptr {
            // The memory input is a Store to the same ptr — return the stored value
            return mem_node.get_input(3);
        }

        let control = self.current_control;
        let mut node = Node::new(NodeKind::Load, loaded_type);
        node.set_inputs(&[control, mem, ptr]);
        let id = self.push_node(&node);
        // Load does NOT update the memory chain (it's a read-only projection)
        id
    }

    /// Create a Store (memory write) node.
    /// `mem` is the memory state before the store.
    /// `ptr` is the pointer (from a New node) to write to.
    /// `value` is the value to store.
    /// Returns a Node whose type is `Type::Memory` and whose NodeId represents
    /// the new memory state after the store.
    /// Automatically updates the current memory at the current control point.
    pub fn create_store(&mut self, mem: NodeId, ptr: NodeId, value: NodeId) -> NodeId {
        let control = self.current_control;
        let mut node = Node::new(NodeKind::Store, Type::Memory);
        node.set_inputs(&[control, mem, ptr, value]);
        let id = self.push_node(&node);
        // Store produces a new memory state — update the memory variable
        self.variables[MEMORY_VAR.as_usize()]
            .current_def
            .insert(control, id);
        id
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::*;

    #[test]
    fn test_arithmetic_constant_folding() {
        let mut builder = IRBuilder::new();

        // Test add constant folding
        let a = Node::const_int(5);
        let b = Node::const_int(3);
        let result = builder.create_add(&a, &b).unwrap();

        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(8));

        // Test sub constant folding
        let result = builder.create_sub(&a, &b).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(2));

        // Test mul constant folding
        let result = builder.create_mul(&a, &b).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(15));

        // Test div constant folding
        let result = builder.create_div(&a, &b).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(1)); // 5/3 = 1 (integer division)
    }

    #[test]
    fn test_arithmetic_peepholes() {
        let mut builder = IRBuilder::new();

        let x = builder.create_param(0, Type::I32);
        let zero = Node::const_int(0);
        let one = Node::const_int(1);

        // Test x + 0 => x
        let result = builder.create_add(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test 0 + x => x
        let result = builder.create_add(&zero, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test x - 0 => x
        let result = builder.create_sub(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test x - x => 0
        let result = builder.create_sub(&x, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(0));

        // Test x * 1 => x
        let result = builder.create_mul(&x, &one).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test 1 * x => x
        let result = builder.create_mul(&one, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test x * 0 => 0
        let result = builder.create_mul(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(0));

        // Test x / 1 => x
        let result = builder.create_div(&x, &one).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);

        // Test x / x => 1
        let result = builder.create_div(&x, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(1));

        // Test 0 / x => 0
        let result = builder.create_div(&zero, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(0));
    }

    #[test]
    fn test_arithmetic_algebraic_peepholes() {
        let mut builder = IRBuilder::new();

        let x = builder.create_param(0, Type::I32);
        let _two = Node::const_int(2);

        // Test x + x => x * 2 (from create_add)
        let result = builder.create_add(&x, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Mul);
        // Should be x * 2
        let mul_rhs = builder.lookup_node(result.get_input(2)); // Right operand is at index 2
        assert_eq!(mul_rhs.t.get_const_int(), Some(2));

        // Skip the complex double negation test for now since it requires more setup
        // The important test (x + x => x * 2) is already working above
    }

    #[test]
    fn test_range_based_constant_folding() {
        // Test that operations on constrained ranges can constant fold

        // Range [5, 5] + Range [3, 3] should fold to const 8
        let constraint_5 = IntConstraint::new(5, 5);
        let constraint_3 = IntConstraint::new(3, 3);
        let type_5 = Type::Int(IntPrim::I64, constraint_5);
        let type_3 = Type::Int(IntPrim::I64, constraint_3);

        let result = type_5.add(&type_3).unwrap();
        assert!(result.is_const());
        assert_eq!(result.get_const_int(), Some(8));

        // Range [1, 10] + Range [5, 5] should give Range [6, 15]
        let range_1_10 = IntConstraint::new(1, 10);
        let const_5 = IntConstraint::new(5, 5);
        let type_range = Type::Int(IntPrim::I64, range_1_10);
        let type_const = Type::Int(IntPrim::I64, const_5);

        let result = type_range.add(&type_const).unwrap();
        match result {
            Type::Int(IntPrim::I64, constraint) => {
                assert_eq!(constraint.min, 6);
                assert_eq!(constraint.max, 15);
            }
            _ => panic!("Expected Int type with constraint"),
        }
    }

    #[test]
    fn test_arithmetic_overflow_handling() {
        // Test that overflow is properly detected and handled

        // i8::MAX + 1 should error when we have constants
        let max_i8 = IntConstraint::new(127, 127); // i8::MAX
        let one = IntConstraint::new(1, 1);
        let max_type = Type::Int(IntPrim::I8, max_i8);
        let one_type = Type::Int(IntPrim::I64, one); // Untyped constant

        let result = max_type.add(&one_type);
        assert!(result.is_err()); // Should error due to provable overflow

        // Full range + full range should not error (runtime case)
        let full_i8 = Type::prim_int(IntPrim::I8);
        let result = full_i8.add(&full_i8);
        assert!(result.is_ok()); // Should fallback to full range, not error
    }

    #[test]
    fn test_division_by_zero_handling() {
        // Test division by zero detection

        let five = IntConstraint::new(5, 5);
        let zero = IntConstraint::new(0, 0);
        let five_type = Type::Int(IntPrim::I64, five);
        let zero_type = Type::Int(IntPrim::I64, zero);

        // 5 / 0 should error
        let result = five_type.div(&zero_type);
        assert!(result.is_err());

        // 5 / range_including_zero should NOT error (possible but not definite division by zero)
        let range_with_zero = IntConstraint::new(-1, 1); // Includes 0
        let range_type = Type::Int(IntPrim::I64, range_with_zero);
        let result = five_type.div(&range_type);
        assert!(result.is_ok()); // Should fallback to full range, not error
    }

    #[test]
    fn test_float_arithmetic() {
        let mut builder = IRBuilder::new();

        // Test float constant folding
        let a = Node::const_float(2.5);
        let b = Node::const_float(1.5);

        let result = builder.create_add(&a, &b).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_float(), Some(4.0));

        let result = builder.create_mul(&a, &b).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_float(), Some(3.75));

        // Test float peepholes
        let x = builder.create_param(0, Type::F64);
        let zero = Node::const_float(0.0);
        let one = Node::const_float(1.0);

        // x * 0.0 => 0.0
        let result = builder.create_mul(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_float(), Some(0.0));

        // x / 1.0 => x
        let result = builder.create_div(&x, &one).unwrap();
        assert_eq!(result.kind, NodeKind::Identity);
        assert_eq!(result.t, x.t);
    }

    #[test]
    fn test_mixed_type_arithmetic_errors() {
        let int_node = Node::const_int(5);
        let float_node = Node::const_float(5.0);
        let bool_node = Node::const_bool(true);

        // Test that mixed types properly error
        let int_type = int_node.t.clone();
        let float_type = float_node.t.clone();
        let bool_type = bool_node.t.clone();

        // int + float should error
        assert!(int_type.add(&float_type).is_err());

        // int + bool should error
        assert!(int_type.add(&bool_type).is_err());

        // float + bool should error
        assert!(float_type.add(&bool_type).is_err());
    }

    #[test]
    fn test_cast_functionality() {
        use crate::constraints::{IntConstraint, UIntConstraint};
        use crate::types::CastKind;
        let mut builder = IRBuilder::new();

        // Test static cast - widening with compatible range
        let narrow_int = Type::Int(IntPrim::I8, IntConstraint::new(10, 20));
        let wide_int = Type::Int(IntPrim::I32, IntConstraint::new(0, 100));
        assert_eq!(narrow_int.cast_kind(&wide_int), CastKind::Static);

        // Test dynamic cast - narrowing
        let wide_range = Type::Int(IntPrim::I32, IntConstraint::new(0, 1000));
        let narrow_range = Type::Int(IntPrim::I32, IntConstraint::new(50, 150));
        assert_eq!(wide_range.cast_kind(&narrow_range), CastKind::Dynamic);

        // Test invalid cast - disjoint ranges
        let range_a = Type::Int(IntPrim::I32, IntConstraint::new(0, 10));
        let range_b = Type::Int(IntPrim::I32, IntConstraint::new(50, 100));
        assert_eq!(range_a.cast_kind(&range_b), CastKind::Invalid);

        // Test cross-type cast (signed to unsigned)
        let signed_positive = Type::Int(IntPrim::I32, IntConstraint::new(0, 100));
        let unsigned_compatible = Type::UInt(UIntPrim::U32, UIntConstraint::new(0, 100));
        assert_eq!(
            signed_positive.cast_kind(&unsigned_compatible),
            CastKind::Static
        );

        let signed_negative = Type::Int(IntPrim::I32, IntConstraint::new(-50, 10));
        let unsigned_range = Type::UInt(UIntPrim::U32, UIntConstraint::new(0, 100));
        assert_eq!(
            signed_negative.cast_kind(&unsigned_range),
            CastKind::Dynamic
        );

        // Test bool casts
        let bool_true = Type::Bool(BoolConstraint::Const(true));
        let bool_false = Type::Bool(BoolConstraint::Const(false));
        let bool_any = Type::Bool(BoolConstraint::Any);

        assert_eq!(bool_true.cast_kind(&bool_any), CastKind::Static); // widening
        assert_eq!(bool_any.cast_kind(&bool_true), CastKind::Dynamic); // narrowing
        assert_eq!(bool_true.cast_kind(&bool_false), CastKind::Invalid); // disjoint

        // Test create_cast method
        let const_5 = Node::const_int(5);
        let target_type = Type::Int(IntPrim::I32, IntConstraint::new(0, 10));

        // This should be a static cast (5 is in range [0, 10])
        let cast_result = builder.create_cast(&const_5, target_type.clone());
        assert!(cast_result.is_ok());
        let cast_node = cast_result.unwrap();
        assert_eq!(cast_node.kind, NodeKind::StaticCast);
        assert_eq!(cast_node.t, target_type);

        // Test invalid cast
        let target_type_invalid = Type::Int(IntPrim::I32, IntConstraint::new(10, 20));
        let cast_result = builder.create_cast(&const_5, target_type_invalid);
        assert!(cast_result.is_err());
        assert_eq!(cast_result.unwrap_err(), IRError::TypeMismatch);
    }

    #[test]
    fn test_union_error_flattening() {
        use crate::types::*;

        // Create two basic types
        let int_type = Type::I32;
        let bool_type = Type::BOOL;

        // Create a union of these types
        let union_type = Type::make_union(vec![int_type.clone(), bool_type.clone()]);

        // Create an Error wrapping this union - this should get flattened
        let error_union = Type::make_error(union_type);

        // Create another union that includes this Error(Union(...))
        let outer_union = Type::make_union(vec![Type::Unit, error_union]);

        // Verify the result: should be Union([Unit, Error(I32), Error(Bool)])
        // with errors at the tail
        if let Type::Union(types, ..) = outer_union {
            assert_eq!(types.len(), 3);
            assert_eq!(types[0], Type::Unit);

            // The errors should be at the tail
            assert!(matches!(types[1], Type::Error(..)));
            assert!(matches!(types[2], Type::Error(..)));

            // Extract inner types from errors to verify they are I32 and Bool
            if let (Type::Error(inner1, ..), Type::Error(inner2, ..)) = (&types[1], &types[2]) {
                let inner_types = vec![(&**inner1).clone(), (&**inner2).clone()];
                // Should contain both original types (order doesn't matter due to sorting)
                assert!(inner_types.contains(&int_type));
                assert!(inner_types.contains(&bool_type));
            } else {
                panic!("Expected Error types at positions 1 and 2");
            }
        } else {
            panic!("Expected Union type, got {:?}", outer_union);
        }
    }

    #[test]
    fn test_outputs_tracking() {
        let mut builder = IRBuilder::new();

        // Create param (node 2) and const (node 3)
        let x = builder.create_param(0, Type::I32);
        let c = Node::const_int(5);
        let c_id = builder.push_node(&c);

        // Add: node 4 = x + c (inputs: 0, x_id, c_id)
        let add_result = builder.create_add(&x, &c).unwrap();
        let add_id = builder.get_node_id(&add_result);

        // Verify outputs: x should have add_id as user, c should have add_id as user
        let x_id = x.get_identity_id().unwrap().as_usize();
        let x_outputs = &builder.node_outputs[x_id];
        assert!(x_outputs.iter().any(|id| id == add_id.0));

        let c_outputs = &builder.node_outputs[c_id.as_usize()];
        assert!(c_outputs.iter().any(|id| id == add_id.0));

        // Create sub: node 5 = add - c
        let sub_result = builder.create_sub(&add_result, &c).unwrap();
        let sub_id = builder.get_node_id(&sub_result);

        // add_id should now have sub_id as a user
        let add_outputs = &builder.node_outputs[add_id.as_usize()];
        assert!(add_outputs.iter().any(|id| id == sub_id.0));

        // c should have both add_id and sub_id as users
        let c_outputs = &builder.node_outputs[c_id.as_usize()];
        assert!(c_outputs.iter().any(|id| id == add_id.0));
        assert!(c_outputs.iter().any(|id| id == sub_id.0));
    }

    #[test]
    fn test_replace_node_outputs() {
        let mut builder = IRBuilder::new();

        let x = builder.create_param(0, Type::I32);
        let c = Node::const_int(5);
        let c_id = builder.push_node(&c);

        // add1: x + 5
        let add1 = builder.create_add(&x, &c).unwrap();
        let add1_id = builder.get_node_id(&add1);

        // Verify c has add1 as a user
        assert!(builder.node_outputs[c_id.as_usize()].iter().any(|id| id == add1_id.0));

        // Now replace c (const 5) with const 0
        let zero = Node::const_int(0);
        let new_c_id = builder.replace_node(c_id, &zero);

        // Old c should have no outputs left
        assert!(builder.node_outputs[c_id.as_usize()].is_empty());

        // New zero const should have add1 as a user
        assert!(builder.node_outputs[new_c_id.as_usize()].iter().any(|id| id == add1_id.0));

        // add1 now references new_c_id instead of c_id
        let add1_inputs: Vec<_> = (0..builder.lookup_node(add1_id).inputs_len())
            .map(|i| builder.lookup_node(add1_id).get_input(i))
            .collect();
        assert!(add1_inputs.contains(&new_c_id));
        assert!(!add1_inputs.contains(&c_id));
    }

    // ── SSA Construction Tests ──

    #[test]
    fn test_ssa_create_variable_and_set_control() {
        let mut builder = IRBuilder::new();

        // Default control starts at Entry
        assert_eq!(builder.current_control(), NodeId(1));

        // Create a variable
        let v = builder.create_variable("x");
        assert_eq!(v, VarId(1)); // VarId(0) is reserved for memory

        // Set control to something else
        builder.set_control(NodeId(1)); // Entry
        assert_eq!(builder.current_control(), NodeId(1));
    }

    #[test]
    fn test_ssa_single_block_read_write() {
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let one = Node::const_int(1);
        let one_id = builder.push_node(&one);

        // Write x = 1 at Entry
        builder.write_variable(v, one_id);

        // Read x at Entry — should return 1
        let result = builder.read_variable(v);
        assert_eq!(result, one_id);
    }

    #[test]
    fn test_ssa_overwrite_within_block() {
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let one = Node::const_int(1);
        let two = Node::const_int(2);
        let one_id = builder.push_node(&one);
        let two_id = builder.push_node(&two);

        // Write x = 1, then overwrite with x = 2
        builder.write_variable(v, one_id);
        builder.write_variable(v, two_id);

        // Should return 2 (most recent)
        assert_eq!(builder.read_variable(v), two_id);
    }

    #[test]
    fn test_ssa_single_predecessor_chain() {
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let five_id = builder.push_node(&five);

        // Write x = 5 at Entry
        builder.write_variable(v, five_id);

        // Entry -> If -> Then (single predecessor chain)
        let cond_true = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond_true);
        let then_ctrl = builder.create_then(if_node);

        // Read at Then — should find x = 5 from Entry (through If)
        builder.set_control(then_ctrl);
        let result = builder.read_variable(v);
        assert_eq!(result, five_id);
    }

    #[test]
    fn test_ssa_if_else_phi_insertion() {
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let ten = Node::const_int(10);
        let five_id = builder.push_node(&five);
        let ten_id = builder.push_node(&ten);

        // Create control flow: Entry -> If -> Then/Else -> Region
        let cond_true = Node::const_bool(true);
        let _cond_id = builder.push_node(&cond_true);
        let if_node = builder.create_if(NodeId(1), &cond_true);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then branch: x = 5
        builder.set_control(then_ctrl);
        builder.write_variable(v, five_id);

        // Else branch: x = 10
        builder.set_control(else_ctrl);
        builder.write_variable(v, ten_id);

        // Merge at Region
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // Read x at Region — should get a Phi
        let result = builder.read_variable(v);
        assert_ne!(result, five_id);
        assert_ne!(result, ten_id);
        assert_ne!(result, NodeId(0)); // Not undef

        // The result should be a Phi node with inputs [region, five, ten]
        let phi_node = builder.lookup_node(result);
        assert_eq!(phi_node.kind, NodeKind::Phi);
        assert_eq!(phi_node.get_input(0), region);
        // Inputs 1 and 2 are the values (order depends on predecessor ordering)
        let val1 = phi_node.get_input(1);
        let val2 = phi_node.get_input(2);
        assert!(
            (val1 == five_id && val2 == ten_id) || (val1 == ten_id && val2 == five_id),
            "Phi operands should be five and ten, got {} and {}",
            val1,
            val2
        );
    }

    #[test]
    fn test_ssa_trivial_phi_elimination() {
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let five_id = builder.push_node(&five);

        // Entry -> If -> Then/Else -> Region
        // Both branches write x = 5 (same value)
        let cond_true = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond_true);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        builder.set_control(then_ctrl);
        builder.write_variable(v, five_id);

        builder.set_control(else_ctrl);
        builder.write_variable(v, five_id);

        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // Read x — should directly return five_id (Phi trivial, same value on both paths)
        let result = builder.read_variable(v);
        assert_eq!(result, five_id);
    }

    #[test]
    fn test_ssa_phi_non_branching_region() {
        // A region with a single predecessor should not create a Phi
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let five_id = builder.push_node(&five);

        builder.write_variable(v, five_id);

        // Create a single-predecessor region (artificial, but valid)
        let region = builder.create_region(&[NodeId(1)]); // Entry -> Region
        builder.set_control(region);

        // Should find five_id through single predecessor chain, no Phi
        let result = builder.read_variable(v);
        assert_eq!(result, five_id);
    }

    #[test]
    fn test_ssa_nested_if_else() {
        // Test a more complex case: outer if with inner if-else
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let one = Node::const_int(1);
        let two = Node::const_int(2);
        let three = Node::const_int(3);
        let one_id = builder.push_node(&one);
        let two_id = builder.push_node(&two);
        let three_id = builder.push_node(&three);

        let cond = Node::const_bool(true);

        // Outer if
        let outer_if = builder.create_if(NodeId(1), &cond);
        let outer_then = builder.create_then(outer_if);
        let outer_else = builder.create_else(outer_if);

        // Outer then: inner if-else
        let inner_if = builder.create_if(outer_then, &cond);
        let inner_then = builder.create_then(inner_if);
        let inner_else = builder.create_else(inner_if);

        builder.set_control(inner_then);
        builder.write_variable(v, one_id);

        builder.set_control(inner_else);
        builder.write_variable(v, two_id);

        let inner_region = builder.create_region(&[inner_then, inner_else]);
        // Outer then ends at inner_region

        // Outer else
        builder.set_control(outer_else);
        builder.write_variable(v, three_id);

        // Final merge
        let outer_region = builder.create_region(&[inner_region, outer_else]);
        builder.set_control(outer_region);

        // Read x — should be a Phi merging the inner Phi's result and 3
        let result = builder.read_variable(v);
        assert_ne!(result, NodeId(0));

        let phi_node = builder.lookup_node(result);
        assert_eq!(phi_node.kind, NodeKind::Phi);
        assert_eq!(phi_node.get_input(0), outer_region);

        // One operand should be the inner Phi, the other should be three_id
        let op1 = phi_node.get_input(1);
        let op2 = phi_node.get_input(2);
        assert!(op1 == three_id || op2 == three_id);

        // The other operand should itself be a Phi (the inner one)
        let inner_candidate = if op1 == three_id { op2 } else { op1 };
        let inner = builder.lookup_node(inner_candidate);
        assert_eq!(inner.kind, NodeKind::Phi);
        assert_eq!(inner.get_input(0), inner_region);
    }

    #[test]
    fn test_ssa_incomplete_phi_on_seal() {
        // Test that Phis placed before a block is sealed get filled on seal()
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let ten = Node::const_int(10);
        let five_id = builder.push_node(&five);
        let ten_id = builder.push_node(&ten);

        // Manually set up: Entry -> Region (with seal deferred)
        // We'll create a Loop (not auto-sealed) to simulate incomplete CFG
        let loop_ctrl = builder.create_loop(NodeId(1)); // Entry -> Loop, NOT sealed

        builder.set_control(loop_ctrl);

        // Write x = 5 at the loop header (not sealed yet)
        builder.write_variable(v, five_id);

        // Now seal the loop — Phis that were placed before sealing get filled
        builder.seal(loop_ctrl);

        // Now write in a branch: add a back-edge
        // Loop already has 1 predecessor (Entry). After several ops, we add back-edge.
        // This simulates the typical loop construction pattern.
        let back_edge = builder.create_region(&[loop_ctrl]); // artificial back-edge
        builder.push_loop_back_edge(loop_ctrl, back_edge);

        // Read at back_edge should find five_id through single-pred chain
        builder.set_control(back_edge);
        let result = builder.read_variable(v);
        assert_eq!(result, five_id); // Should find five from sealed loop

        // Now write ten at back_edge and seal the back_edge region
        builder.write_variable(v, ten_id);
        // Re-seal the loop header after adding back-edge...
        // Actually, once a block is sealed we can't add more preds.
        // The back_edge is a separate region, so this is fine.
    }

    #[test]
    fn test_ssa_chained_trivial_phi_elimination() {
        // Test that eliminating one trivial Phi cascades to eliminate users
        let mut builder = IRBuilder::new();

        let v0 = builder.create_variable("a");
        let v1 = builder.create_variable("b");
        let five = Node::const_int(5);
        let five_id = builder.push_node(&five);

        // Create: Entry -> Region1 -> Region2
        // Write a = 5 at Entry
        // Then try reading a and b at Region2

        builder.write_variable(v0, five_id);

        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Both branches write b = a (same as reading a and re-writing)
        builder.set_control(then_ctrl);
        let a_val_then = builder.read_variable(v0);
        builder.write_variable(v1, a_val_then);

        builder.set_control(else_ctrl);
        let a_val_else = builder.read_variable(v0);
        builder.write_variable(v1, a_val_else);

        // Merge at Region
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // Read b — should be a Phi merging the two same values
        // The Phi should be trivial since both operands are five_id!
        let b_val = builder.read_variable(v1);
        assert_eq!(b_val, five_id);
    }

    #[test]
    fn test_ssa_undefined_variable() {
        // Reading a variable that was never written should return Unreachable (node 0)
        let mut builder = IRBuilder::new();

        let v = builder.create_variable("uninit");

        // Read at Entry with no prior write
        let result = builder.read_variable(v);
        assert_eq!(result, NodeId(0)); // Should return Unreachable (node 0)
    }

    #[test]
    fn test_ssa_replace_all_uses() {
        let mut builder = IRBuilder::new();

        // Create: param + const, then x = add(param, const)
        let x = builder.create_param(0, Type::I32);
        let c = Node::const_int(5);
        let c_id = builder.push_node(&c);
        let add = builder.create_add(&x, &c).unwrap();
        let add_id = builder.get_node_id(&add);

        // c has add as a user
        assert!(builder.node_outputs[c_id.as_usize()].iter().any(|id| id == add_id.0));

        // Replace all uses of c (const 5) with another const
        let new_c = Node::const_int(10);
        let new_c_id = builder.push_node(&new_c);
        builder.replace_all_uses(c_id, new_c_id);

        // Old c has no users
        assert!(builder.node_outputs[c_id.as_usize()].is_empty());

        // New c has add as a user
        assert!(builder.node_outputs[new_c_id.as_usize()].iter().any(|id| id == add_id.0));

        // Add now uses new_c_id
        let add_inputs: Vec<_> = (0..builder.lookup_node(add_id).inputs_len())
            .map(|i| builder.lookup_node(add_id).get_input(i))
            .collect();
        assert!(add_inputs.contains(&new_c_id));
        assert!(!add_inputs.contains(&c_id));
    }

    #[test]
    fn test_ssa_multiple_vars_at_merge() {
        let mut builder = IRBuilder::new();
        let x = builder.create_variable("x");
        let y = builder.create_variable("y");

        let one = Node::const_int(1);
        let two = Node::const_int(2);
        let three = Node::const_int(3);
        let four = Node::const_int(4);
        let one_id = builder.push_node(&one);
        let two_id = builder.push_node(&two);
        let three_id = builder.push_node(&three);
        let four_id = builder.push_node(&four);

        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then: x=1, y=2
        builder.set_control(then_ctrl);
        builder.write_variable(x, one_id);
        builder.write_variable(y, two_id);

        // Else: x=3, y=4
        builder.set_control(else_ctrl);
        builder.write_variable(x, three_id);
        builder.write_variable(y, four_id);

        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // Read both vars — each should have its own Phi
        let x_val = builder.read_variable(x);
        let y_val = builder.read_variable(y);

        assert_ne!(x_val, NodeId(0));
        assert_ne!(y_val, NodeId(0));
        assert_ne!(x_val, y_val); // Different Phis

        // x's Phi should merge 1 and 3
        let x_phi = builder.lookup_node(x_val);
        assert_eq!(x_phi.kind, NodeKind::Phi);
        assert!(x_phi.get_input(1) == one_id || x_phi.get_input(2) == one_id);
        assert!(x_phi.get_input(1) == three_id || x_phi.get_input(2) == three_id);

        // y's Phi should merge 2 and 4
        let y_phi = builder.lookup_node(y_val);
        assert_eq!(y_phi.kind, NodeKind::Phi);
        assert!(y_phi.get_input(1) == two_id || y_phi.get_input(2) == two_id);
        assert!(y_phi.get_input(1) == four_id || y_phi.get_input(2) == four_id);
    }

    #[test]
    fn test_ssa_partial_definition_at_merge() {
        // Variable defined on one branch, falls through to outer def on the other
        let mut builder = IRBuilder::new();
        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let ten = Node::const_int(10);
        let five_id = builder.push_node(&five);
        let ten_id = builder.push_node(&ten);

        // Write x = 5 at Entry
        builder.write_variable(v, five_id);

        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then branch: overwrites x = 10
        builder.set_control(then_ctrl);
        builder.write_variable(v, ten_id);

        // Else branch: no write — falls through to Entry definition (x = 5)

        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // Read x — should get a Phi merging 10 (from Then) and 5 (from Entry via Else)
        let result = builder.read_variable(v);
        assert_ne!(result, NodeId(0));

        let phi = builder.lookup_node(result);
        assert_eq!(phi.kind, NodeKind::Phi);
        // One operand is ten_id, the other is five_id
        let op1 = phi.get_input(1);
        let op2 = phi.get_input(2);
        assert!(
            (op1 == five_id && op2 == ten_id) || (op1 == ten_id && op2 == five_id),
            "Phi operands should be 5 and 10, got {} and {}",
            op1,
            op2
        );
    }

    #[test]
    fn test_ssa_phi_with_self_reference_collapses() {
        // A Phi that only references itself and/or undef should collapse to undef.
        // This occurs when a variable is read inside a loop before its first write.
        let mut builder = IRBuilder::new();
        let v = builder.create_variable("x");

        // Create a loop (not sealed), read x — creates an incomplete Phi
        let loop_ctrl = builder.create_loop(NodeId(1));
        builder.set_control(loop_ctrl);
        let _phi_at_loop = builder.read_variable(v);

        // Add a back-edge and seal: the Phi gets operands filled.
        // Since neither Entry nor back-edge define x, it collapses to NodeId(0).
        let back_edge = builder.create_region(&[loop_ctrl]);
        builder.push_loop_back_edge(loop_ctrl, back_edge);
        builder.seal(loop_ctrl);

        // After sealing, the incomplete Phi should have been filled and collapsed.
        // The current def at loop_ctrl should be NodeId(0).
        builder.set_control(loop_ctrl);
        assert_eq!(builder.read_variable(v), NodeId(0));
    }

    #[test]
    fn test_ssa_seal_with_no_incomplete_phis() {
        // Sealing a block with no incomplete Phis should be a no-op
        let mut builder = IRBuilder::new();

        let region = builder.create_region(&[NodeId(1)]);
        // region is auto-sealed, no Phis were ever created for it — no-op
        builder.seal(region);

        // Should still be sealed
        // Read a variable that has a def at Entry — should work through region
        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let five_id = builder.push_node(&five);
        builder.write_variable(v, five_id);

        builder.set_control(region);
        assert_eq!(builder.read_variable(v), five_id);
    }

    #[test]
    fn test_ssa_replace_all_uses_cleans_up_inputs() {
        // Verify that replace_all_uses removes old_id from its inputs' output lists
        let mut builder = IRBuilder::new();

        let x = builder.create_param(0, Type::I32);
        let _x_id = x.get_identity_id().unwrap();
        let c = Node::const_int(5);
        let c_id = builder.push_node(&c);

        // add uses both x and c
        let add = builder.create_add(&x, &c).unwrap();
        let add_id = builder.get_node_id(&add);

        // add is a user of c
        assert!(builder.node_outputs[c_id.as_usize()].iter().any(|id| id == add_id.0));

        // c is an input of add, so add's input list includes c_id
        // Now replace all uses of c
        let new_c = Node::const_int(10);
        let new_c_id = builder.push_node(&new_c);
        builder.replace_all_uses(c_id, new_c_id);

        // c should no longer have add as a user (stale edge cleaned up)
        assert!(builder.node_outputs[c_id.as_usize()].is_empty());

        // new_c should have add as a user
        assert!(builder.node_outputs[new_c_id.as_usize()].iter().any(|id| id == add_id.0));
    }

    // ── Memory Tests ──

    #[test]
    fn test_memory_node_created() {
        let builder = IRBuilder::new();
        // Node 0 = Unreachable, Node 1 = Entry, Node 2 = Memory
        assert_eq!(builder.nodes[2].kind, NodeKind::Memory);
        assert_eq!(builder.nodes[2].t, Type::Memory);
        assert_eq!(builder.nodes[2].inputs_len(), 0);
    }

    #[test]
    fn test_memory_current_memory_at_entry() {
        let mut builder = IRBuilder::new();
        // At Entry, the current memory should be the Memory node (node 2)
        assert_eq!(builder.get_current_memory(), NodeId(2));
    }

    #[test]
    fn test_memory_set_current_memory() {
        let mut builder = IRBuilder::new();
        let mem_id = NodeId(2); // initial Memory node
        assert_eq!(builder.get_current_memory(), mem_id);

        // Set a different memory (e.g., a Store or New node id)
        builder.set_current_memory(NodeId(42));
        assert_eq!(builder.get_current_memory(), NodeId(42));
    }

    #[test]
    fn test_memory_single_predecessor_chain() {
        let mut builder = IRBuilder::new();

        // Entry -> If -> Then (single predecessor chain)
        let cond_true = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond_true);
        let then_ctrl = builder.create_then(if_node);

        // Memory at Then should propagate through chain: Entry -> If -> Then
        builder.set_control(then_ctrl);
        let mem = builder.get_current_memory();
        assert_eq!(mem, NodeId(2)); // Should find the initial Memory node
    }

    #[test]
    fn test_memory_new_creates_node_and_updates_chain() {
        let mut builder = IRBuilder::new();

        let mem_id = builder.get_current_memory();
        assert_eq!(mem_id, NodeId(2)); // initial Memory

        // Allocate: create a new Foo object
        let foo_type = Type::make_tuple(vec![Type::I32, Type::I32]);
        let new_id = builder.create_new(mem_id, foo_type.clone());

        // The New node should be a node with kind New and type foo_type
        assert_eq!(builder.nodes[new_id.as_usize()].kind, NodeKind::New);
        assert_eq!(builder.nodes[new_id.as_usize()].t, foo_type);

        // New takes control and mem as inputs
        assert_eq!(builder.nodes[new_id.as_usize()].get_input(0), NodeId(1)); // control = Entry
        assert_eq!(builder.nodes[new_id.as_usize()].get_input(1), NodeId(2)); // mem = Memory root

        // Current memory should now be the New node
        assert_eq!(builder.get_current_memory(), new_id);
    }

    #[test]
    fn test_memory_store_creates_node_and_updates_chain() {
        let mut builder = IRBuilder::new();

        let mem_id = builder.get_current_memory(); // Memory root (node 2)
        let foo_type = Type::make_tuple(vec![Type::I32, Type::I32]);

        // Allocate first
        let ptr = builder.create_new(mem_id, foo_type);
        // Current memory is now the New node

        // Store: write a value into the allocated object
        let val = Node::const_int(42);
        let val_id = builder.push_node(&val);
        let current_mem = builder.get_current_memory(); // Should be New
        let store_id = builder.create_store(current_mem, ptr, val_id);

        // The Store node should have kind Store and type Memory
        assert_eq!(builder.nodes[store_id.as_usize()].kind, NodeKind::Store);
        assert_eq!(builder.nodes[store_id.as_usize()].t, Type::Memory);

        // Store inputs: [control, mem, ptr, value]
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(0), NodeId(1)); // control = Entry
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(1), current_mem); // mem = New node
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(2), ptr);
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(3), val_id);

        // Current memory should now be the Store node
        assert_eq!(builder.get_current_memory(), store_id);
    }

    #[test]
    fn test_memory_load_creates_node() {
        let mut builder = IRBuilder::new();

        let mem_id = builder.get_current_memory(); // Memory root (node 2)
        let foo_type = Type::make_tuple(vec![Type::I32, Type::I32]);

        // Allocate
        let ptr = builder.create_new(mem_id, foo_type);

        // Load from the allocation
        let loaded_type = Type::I32;
        let load_mem = builder.get_current_memory(); // New node
        let load_id = builder.create_load(load_mem, ptr, loaded_type.clone());

        // The Load node should have kind Load and type I32
        assert_eq!(builder.nodes[load_id.as_usize()].kind, NodeKind::Load);
        assert_eq!(builder.nodes[load_id.as_usize()].t, loaded_type);

        // Load inputs: [control, mem, ptr]
        assert_eq!(builder.nodes[load_id.as_usize()].get_input(0), NodeId(1)); // control = Entry
        assert_eq!(builder.nodes[load_id.as_usize()].get_input(1), load_mem); // mem = New
        assert_eq!(builder.nodes[load_id.as_usize()].get_input(2), ptr);

        // Load does NOT update the memory chain — memory should still be the New node
        assert_eq!(builder.get_current_memory(), load_mem);
    }

    #[test]
    fn test_memory_chain_new_then_store_then_load() {
        let mut builder = IRBuilder::new();

        let mem_root = builder.get_current_memory(); // Memory root (node 2)
        let obj_type = Type::make_tuple(vec![Type::I32]);

        // Chain: Memory -> New -> Store
        let ptr = builder.create_new(mem_root, obj_type);
        let store_mem = builder.get_current_memory(); // should be New
        let val = Node::const_int(99);
        let val_id = builder.push_node(&val);
        let store_id = builder.create_store(store_mem, ptr, val_id);

        // Now load: uses Store's memory state — Load-Store forwarding returns val_id directly
        let load_mem = builder.get_current_memory(); // should be Store
        let load_result = builder.create_load(load_mem, ptr, Type::I32);
        assert_eq!(load_result, val_id, "Load should forward to stored value");

        // Verify the chain edges (Memory -> New -> Store)
        assert_eq!(builder.nodes[ptr.as_usize()].get_input(1), mem_root);
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(1), store_mem);
        assert_eq!(builder.nodes[store_id.as_usize()].get_input(3), val_id);

        // Memory chain: 2 (Memory) -> ptr (New) -> store_id (Store)
        assert_eq!(mem_root, NodeId(2));
        assert_eq!(store_mem, ptr);
        assert_eq!(load_mem, store_id);
    }

    #[test]
    fn test_memory_phi_at_merge() {
        let mut builder = IRBuilder::new();

        let mem_root = builder.get_current_memory(); // Memory root (node 2)
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        // Entry -> If -> Then/Else -> Region
        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then branch: allocate and store
        builder.set_control(then_ctrl);
        let then_mem = builder.get_current_memory(); // should propagate from Entry
        assert_eq!(then_mem, mem_root);
        let ptr_then = builder.create_new(then_mem, obj_type.clone());
        let val1 = Node::const_int(1);
        let val1_id = builder.push_node(&val1);
        let then_before_store = builder.get_current_memory();
        builder.create_store(then_before_store, ptr_then, val1_id);
        let then_final_mem = builder.get_current_memory(); // the Store

        // Else branch: just allocate (different ptr name but same type)
        builder.set_control(else_ctrl);
        let else_mem = builder.get_current_memory(); // should propagate from Entry
        assert_eq!(else_mem, mem_root);
        let ptr_else = builder.create_new(else_mem, obj_type);
        let val2 = Node::const_int(2);
        let val2_id = builder.push_node(&val2);
        let else_before_store = builder.get_current_memory();
        builder.create_store(else_before_store, ptr_else, val2_id);
        let else_final_mem = builder.get_current_memory(); // the Store

        // Merge at Region — memory should be different on both paths, so a memory Phi is needed
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        let merged_mem = builder.get_current_memory();

        // Should be a memory Phi (since Then and Else have different memory states)
        let phi_node = builder.lookup_node(merged_mem);
        assert_eq!(phi_node.kind, NodeKind::Phi);
        assert_eq!(phi_node.t, Type::Memory);
        assert_eq!(phi_node.get_input(0), region);

        // The Phi should merge the two final memory states
        let op1 = phi_node.get_input(1);
        let op2 = phi_node.get_input(2);
        assert!(
            (op1 == then_final_mem && op2 == else_final_mem) || (op1 == else_final_mem && op2 == then_final_mem),
            "Memory Phi should merge Then and Else memory states, got {} and {}",
            op1,
            op2
        );
    }

    #[test]
    fn test_memory_trivial_phi_elimination() {
        let mut builder = IRBuilder::new();

        // Entry -> If -> Then/Else -> Region
        // Both branches have the SAME memory (no stores/allocations), so memory Phi should be trivial
        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Neither branch modifies memory — both reach the same Memory root
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        // No memory Phi needed — both paths reach the same memory (Memory root = node 2)
        let mem = builder.get_current_memory();
        assert_eq!(mem, NodeId(2)); // Directly the Memory root, no Phi
    }

    #[test]
    fn test_memory_partial_definition_at_merge() {
        // Memory modified on one branch only, passes through on the other
        let mut builder = IRBuilder::new();

        let mem_root = builder.get_current_memory();

        // Entry -> If -> Then/Else -> Region
        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then branch: allocate (changes memory)
        builder.set_control(then_ctrl);
        let obj_type = Type::make_record(vec![("x", Type::I32)]);
        let then_ctrl_mem = builder.get_current_memory();
        let _ptr_then = builder.create_new(then_ctrl_mem, obj_type);
        let then_mem = builder.get_current_memory(); // the New node

        // Else branch: NO memory modification — still has the Memory root
        builder.set_control(else_ctrl);
        // Current memory here should propagate through Entry -> If -> Else, landing on Memory root
        assert_eq!(builder.get_current_memory(), mem_root);

        // Merge at Region
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);

        let merged_mem = builder.get_current_memory();

        // Should be a Phi: merges 'then_mem' (New) and mem_root (Memory) since one path changed it
        let phi_node = builder.lookup_node(merged_mem);
        assert_eq!(phi_node.kind, NodeKind::Phi);
        assert_eq!(phi_node.t, Type::Memory);

        let op1 = phi_node.get_input(1);
        let op2 = phi_node.get_input(2);
        assert!(
            (op1 == then_mem && op2 == mem_root) || (op1 == mem_root && op2 == then_mem),
            "Memory Phi should merge then_mem ({}) and mem_root ({}), got {} and {}",
            then_mem, mem_root, op1, op2
        );
    }

    #[test]
    fn test_memory_load_after_store_reads_same() {
        // In Sea of Nodes, Load-Store forwarding means a Load immediately after
        // a Store to the same ptr returns the stored value directly (no Load node).
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let point_type = Type::make_record(vec![("x", Type::I32)]);

        // Allocate, store x=42, then load x
        let ptr = builder.create_new(mem, point_type);
        let val = Node::const_int(42);
        let val_id = builder.push_node(&val);
        let store_before = builder.get_current_memory();
        let store_mem = builder.create_store(store_before, ptr, val_id);

        // Load from same ptr, using Store as memory — forwarded to stored value
        let load_result = builder.create_load(store_mem, ptr, Type::I32);
        assert_eq!(load_result, val_id, "Load should be forwarded to the stored value");
    }

    #[test]
    fn test_memory_incomplete_phi_on_seal() {
        // Test that memory Phis placed before a block is sealed get filled on seal()
        let mut builder = IRBuilder::new();

        // Create a Loop (not auto-sealed) to simulate incomplete CFG
        let loop_ctrl = builder.create_loop(NodeId(1)); // Entry -> Loop, NOT sealed
        builder.set_control(loop_ctrl);

        // Read memory at unsealed loop — creates an incomplete memory Phi
        let _mem_at_loop = builder.get_current_memory();

        // The incomplete_phis should have an entry for MEMORY_VAR at this control
        assert!(builder.incomplete_phis.contains_key(&(MEMORY_VAR, loop_ctrl)));

        // Create back-edge and seal
        let back_edge = builder.create_region(&[loop_ctrl]);
        builder.push_loop_back_edge(loop_ctrl, back_edge);
        builder.seal(loop_ctrl);

        // After sealing, the incomplete Phi should have been filled
        assert!(!builder.incomplete_phis.contains_key(&(MEMORY_VAR, loop_ctrl)));

        // Memory at loop should be resolved (either Memory root, or a trivial Phi collapsed)
        let mem = builder.get_current_memory();
        assert_eq!(mem, NodeId(2)); // Both predecessors reach Memory root — Phi collapsed
    }

    #[test]
    fn test_memory_load_store_forwarding() {
        // Load immediately after Store to same ptr → return stored value directly
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        let ptr = builder.create_new(mem, obj_type);
        let store_before = builder.get_current_memory();
        let val = Node::const_int(42);
        let val_id = builder.push_node(&val);
        let store = builder.create_store(store_before, ptr, val_id);

        // Load from same ptr using Store as memory → should forward to val_id
        let load_result = builder.create_load(store, ptr, Type::I32);
        assert_eq!(load_result, val_id, "Load should forward to Store's value");
    }

    #[test]
    fn test_memory_load_without_store_creates_node() {
        // Load from New without any prior Store → should create a Load node
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        let ptr = builder.create_new(mem, obj_type);
        let new_mem = builder.get_current_memory(); // the New node

        let load_id = builder.create_load(new_mem, ptr, Type::I32);
        assert_ne!(load_id, NodeId(0));
        let load_node = builder.lookup_node(load_id);
        assert_eq!(load_node.kind, NodeKind::Load);
    }

    #[test]
    fn test_memory_load_store_diff_ptr_no_forward() {
        // Store to ptr1, Load from ptr2 → should NOT forward
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj1_type = Type::make_record(vec![("x", Type::I32)]);
        let obj2_type = Type::make_record(vec![("x", Type::I32)]);

        let ptr1 = builder.create_new(mem, obj1_type);
        let mem_after_new1 = builder.get_current_memory();
        let ptr2 = builder.create_new(mem_after_new1, obj2_type);

        // Store to ptr1
        let val = Node::const_int(7);
        let val_id = builder.push_node(&val);
        let store_before = builder.get_current_memory();
        let store = builder.create_store(store_before, ptr1, val_id);

        // Load from ptr2 (different ptr) — should NOT forward
        let mem_after_store = builder.get_current_memory();
        let load_id = builder.create_load(mem_after_store, ptr2, Type::I32);
        let load_node = builder.lookup_node(load_id);
        assert_eq!(
            load_node.kind,
            NodeKind::Load,
            "Load from different ptr should create a Load node"
        );
    }

    #[test]
    fn test_memory_load_forward_from_last_store() {
        // Two Stores to the same ptr, then Load → should forward from last Store
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        let ptr = builder.create_new(mem, obj_type);
        let mem_after_new = builder.get_current_memory();

        // First store: x = 7
        let val1 = Node::const_int(7);
        let val1_id = builder.push_node(&val1);
        let store1 = builder.create_store(mem_after_new, ptr, val1_id);

        // Second store: x = 99 (overwrites)
        let val2 = Node::const_int(99);
        let val2_id = builder.push_node(&val2);
        let store2 = builder.create_store(store1, ptr, val2_id);

        // Load — should forward to 99 (the last store's value)
        let load_result = builder.create_load(store2, ptr, Type::I32);
        assert_eq!(
            load_result, val2_id,
            "Load should forward to the most recent Store's value"
        );
    }

    #[test]
    fn test_memory_load_store_via_phi_no_forward() {
        // When memory reaches Load through a Phi, we can't forward
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        // Entry -> If -> Then/Else -> Region
        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        // Then: allocate and store 10
        builder.set_control(then_ctrl);
        let ptr = builder.create_new(mem, obj_type.clone());
        let then_mem_after_new = builder.get_current_memory();
        let val = Node::const_int(10);
        let val_id = builder.push_node(&val);
        let then_store = builder.create_store(then_mem_after_new, ptr, val_id);

        // Else: just allocate (no store), different ptr
        builder.set_control(else_ctrl);
        let _ptr_else = builder.create_new(mem, obj_type);
        let else_mem = builder.get_current_memory();

        // Merge at Region — memory Phi merges Store and New
        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);
        let merged_mem = builder.get_current_memory();

        // Load from ptr using merged memory (a Phi) — should NOT forward
        let load_id = builder.create_load(merged_mem, ptr, Type::I32);
        let load_node = builder.lookup_node(load_id);
        assert_eq!(
            load_node.kind,
            NodeKind::Load,
            "Load via memory Phi should create a Load node"
        );
    }

    #[test]
    fn test_memory_load_store_skips_output_edges() {
        // When Load-Store forwarding fires, no extra output edges should appear
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);

        let ptr = builder.create_new(mem, obj_type);
        let store_before = builder.get_current_memory();
        let val = Node::const_int(42);
        let val_id = builder.push_node(&val);
        let store = builder.create_store(store_before, ptr, val_id);

        // Track outputs of Store and value before Load
        let store_outputs_before = builder.node_outputs[store.as_usize()].len();
        let val_outputs_before = builder.node_outputs[val_id.as_usize()].len();

        // Load that gets forwarded — should not add output edges
        let _load_result = builder.create_load(store, ptr, Type::I32);

        // Store's outputs should not have changed (no Load node created)
        assert_eq!(
            builder.node_outputs[store.as_usize()].len(),
            store_outputs_before,
            "Store should not gain output edges from forwarded Load"
        );

        // Value's outputs should also not have changed
        assert_eq!(
            builder.node_outputs[val_id.as_usize()].len(),
            val_outputs_before,
            "Value should not gain output edges from forwarded Load"
        );
    }

    #[test]
    fn test_memory_output_tracking() {
        // Verify that memory operations properly track output edges
        let mut builder = IRBuilder::new();

        let mem = builder.get_current_memory();
        let obj_type = Type::make_tuple(vec![Type::I32]);

        // Allocate: New consumes Memory, produces ptr/NewMem
        let ptr = builder.create_new(mem, obj_type);

        // Memory root should have New as a user
        assert!(builder.node_outputs[mem.as_usize()]
            .iter()
            .any(|id| id == ptr.0));

        // Allocate a second object BEFORE the store so we can use the store's
        // memory with a different pointer (no forwarding).
        let obj2_type = Type::make_tuple(vec![Type::I32]);
        let current_mem = builder.get_current_memory();
        let ptr2 = builder.create_new(current_mem, obj2_type);
        // Now current memory is the 2nd New (ptr2). The store will write to ptr.
        let store_mem = builder.get_current_memory(); // 2nd New
        let val = Node::const_int(7);
        let val_id = builder.push_node(&val);
        let store = builder.create_store(store_mem, ptr, val_id);

        // New (ptr) should have Store as a user (ptr is the first New)
        assert!(builder.node_outputs[ptr.as_usize()]
            .iter()
            .any(|id| id == store.0));

        // Load from ptr2 (not ptr) using Store's memory — different ptrs, no forwarding.
        let load = builder.create_load(store, ptr2, Type::I32);
        assert!(builder.node_outputs[store.as_usize()]
            .iter()
            .any(|id| id == load.0));
    }
}
