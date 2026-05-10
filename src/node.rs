use crate::compact_vec::{CompactVec, CompactVecData, CompactVecState, CompactVecStateU16};
use crate::types::Type;

// ── NodeId ──

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

// ── VarId ──

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

// ── I/O vector type aliases ──

type InputsVecData = CompactVecData<16, 8, 4>;
pub(crate) type OutputsVec = CompactVec<16, 8, 4>;

// ── IRError ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IRError {
    TypeMismatch,
    IntegerOverflow,
    DivisionByZero,
    InvalidPrimitiveCoercion,
}

// ── NodeKind ──

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Unreachable,

    /// Fake node containing the ID of the real node stored in the nodes array.
    /// Lets us use the Node value-based interface even for nodes with identity.
    /// Inputs: none
    Identity,

    /// Function parameter (index stored in data.param_index).
    /// Inputs: none
    Param,

    /// Constant value (value encoded in the type).
    /// Inputs: none
    Const,

    /// Data merge node.
    /// Inputs: control (region), value per predecessor
    Phi,

    // Control nodes
    Entry,  // Entry control node (aka start). Inputs: none
    If,     // Inputs: control, predicate
    Then,   // Then branch projection. Inputs: control (if)
    Else,   // Else branch projection. Inputs: control (if)
    Region, // Control merge node. Inputs: control per predecessor
    Loop,   // Special control merge node for loops. Inputs: control per predecessor

    // Memory nodes
    Memory, // Initial mem state. Root of all mem chains. Value is mem. Inputs: none
    New,    // Allocate new object of type matching node type. Value is ptr. Inputs: mem
    Load,   // Load from ptr. Value is loaded data, but can polymorphically act as mem. Inputs: mem, ptr (from New), offset
    Store,  // Store to ptr. Value is mem. Inputs: mem, ptr (from New), value

    // Unary operations — inputs: value
    Neg,
    Not,

    // Binary operations — inputs: lhs, rhs
    Add,
    Sub,
    Mul,
    Div,

    // Cast operations — inputs: control, value
    StaticCast,  // Safe cast, no runtime check needed
    DynamicCast, // Cast requiring runtime type check
}

impl NodeKind {
    pub fn is_pure(self) -> bool {
        use NodeKind::*;
        matches!(
            self,
            Const | Neg | Not | Add | Sub | Mul | Div | StaticCast | DynamicCast
        )
    }

    /// Returns true for node kinds whose value might simplify when their inputs change.
    /// This includes pure nodes (peepholes, constant folding) and Phis (trivial elimination).
    /// These are the kinds that get enqueued on the worklist in replace_all_uses.
    pub fn can_idealize(self) -> bool {
        self.is_pure() || self == NodeKind::Phi
    }
}

// ── NodeData (union) ──

#[derive(Copy, Clone)]
pub(crate) union NodeData {
    /// Compact vector for inputs
    inputs: InputsVecData,
    /// Parameter index for Param nodes
    pub(crate) param_index: usize,
    /// Real node ID for Identity nodes
    pub(crate) identity_id: NodeId,
}

impl std::fmt::Debug for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeData").finish_non_exhaustive()
    }
}

// ── Node ──

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    _flags: u8,
    inputs_state: CompactVecStateU16,
    _pad: u32,
    pub t: Type,
    pub(crate) data: NodeData,
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

    // ── Named accessors ──
    // Every node that depends on control stores it at index 0.
    // Kind-specific operands start at index 1.

    /// Control dependency (index 0). Valid for all control-dependent nodes.
    #[inline]
    pub fn ctrl(&self) -> NodeId {
        self.get_input(0)
    }

    /// Left-hand side of a binary operation (Add, Sub, Mul, Div).
    #[inline]
    pub fn lhs(&self) -> NodeId {
        debug_assert!(matches!(
            self.kind,
            NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Div
        ));
        self.get_input(1)
    }

    /// Right-hand side of a binary operation (Add, Sub, Mul, Div).
    #[inline]
    pub fn rhs(&self) -> NodeId {
        debug_assert!(matches!(
            self.kind,
            NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Div
        ));
        self.get_input(2)
    }

    /// Memory chain input (New, Load, Store).
    #[inline]
    pub fn memory(&self) -> NodeId {
        debug_assert!(matches!(self.kind, NodeKind::New | NodeKind::Load | NodeKind::Store));
        self.get_input(1)
    }

    /// Pointer operand (Load, Store).
    #[inline]
    pub fn ptr(&self) -> NodeId {
        debug_assert!(matches!(self.kind, NodeKind::Load | NodeKind::Store));
        self.get_input(2)
    }

    /// Value being stored (Store only).
    #[inline]
    pub fn store_value(&self) -> NodeId {
        debug_assert_eq!(self.kind, NodeKind::Store);
        self.get_input(3)
    }

    /// Single value operand (Neg, Not, StaticCast, DynamicCast).
    #[inline]
    pub fn value(&self) -> NodeId {
        debug_assert!(matches!(
            self.kind,
            NodeKind::Neg | NodeKind::Not | NodeKind::StaticCast | NodeKind::DynamicCast
        ));
        self.get_input(1)
    }

    /// Condition operand (If).
    #[inline]
    pub fn predicate(&self) -> NodeId {
        debug_assert_eq!(self.kind, NodeKind::If);
        self.get_input(1)
    }

    /// Control region merge point (Phi).
    #[inline]
    pub fn region(&self) -> NodeId {
        debug_assert_eq!(self.kind, NodeKind::Phi);
        self.get_input(0)
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
        // Hash discriminant-specific union data
        match self.kind {
            NodeKind::Param => unsafe { self.data.param_index.hash(state) },
            NodeKind::Identity => unsafe { self.data.identity_id.hash(state) },
            _ => {}
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
        // Compare discriminant-specific union data
        match self.kind {
            NodeKind::Param => unsafe { self.data.param_index == other.data.param_index },
            NodeKind::Identity => unsafe { self.data.identity_id == other.data.identity_id },
            _ => true,
        }
    }
}

impl Eq for Node {}

impl Drop for Node {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.data.inputs.drop_heap(&self.inputs_state);
        }
    }
}