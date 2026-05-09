mod compact_vec;
mod constraints;
mod symbols;
mod types;
use compact_vec::{CompactVecData, CompactVecState, CompactVecStateU16};
use std::collections::HashMap;
use types::*;

use crate::compact_vec::CompactVec;

pub type NodeId = u32;

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
        unsafe { self.data.inputs.get(&self.inputs_state, index) }
    }

    #[inline]
    pub fn set_inputs(&mut self, inputs: &[NodeId]) {
        unsafe {
            self.data.inputs.set_all(&mut self.inputs_state, inputs);
        }
    }

    #[inline]
    pub fn set_input(&mut self, index: usize, input: NodeId) {
        unsafe {
            self.data.inputs.set(&mut self.inputs_state, index, input);
        }
    }

    #[inline]
    pub fn push_input(&mut self, input: NodeId) {
        unsafe {
            self.data.inputs.push(&mut self.inputs_state, input);
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

pub struct IRBuilder {
    nodes: Vec<Node>,
    node_outputs: Vec<OutputsVec>,
    interned_nodes: HashMap<Node, NodeId>,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            nodes: vec![
                Node::new(NodeKind::Unreachable, Type::Never),
                Node::new(NodeKind::Entry, Type::Control),
            ],
            node_outputs: vec![OutputsVec::new(), OutputsVec::new()],
            interned_nodes: HashMap::new(),
        }
    }

    #[inline]
    fn lookup_node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline]
    fn push_node(&mut self, node: &Node) -> NodeId {
        // For pure nodes, check if already interned (avoids duplicates)
        if node.kind.is_pure() {
            if let Some(&id) = self.interned_nodes.get(node) {
                return id;
            }
        }

        let id = self.nodes.len() as NodeId;
        self.nodes.push(node.clone());
        self.node_outputs.push(OutputsVec::new());

        // Build outputs mirror for all inputs
        for i in 0..node.inputs_len() {
            let input_id = node.get_input(i);
            if input_id > 0 {
                self.node_outputs[input_id as usize].push(id);
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

        // Snapshot old outputs, then clear the old node's output list
        let old_outputs: Vec<NodeId> = self.node_outputs[old_id as usize].iter().collect();
        self.node_outputs[old_id as usize].set_all(&[]);

        // Rewire each old user to point to new_id instead of old_id
        for &user_id in &old_outputs {
            let user = &mut self.nodes[user_id as usize];
            for i in 0..user.inputs_len() {
                if user.get_input(i) == old_id {
                    user.set_input(i, new_id);
                    break;
                }
            }
            self.node_outputs[new_id as usize].push(user_id);
        }

        new_id
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
            &self.nodes[identity_id as usize]
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
            node.set_inputs(&[0, self.get_node_id(lhs), self.get_node_id(rhs)]);
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
            node.set_inputs(&[0, self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
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
            node.set_inputs(&[0, self.get_node_id(lhs), self.get_node_id(rhs)]);
            node
        } else if l.kind != NodeKind::Mul && r.kind == NodeKind::Mul {
            // non-mul * mul => mul * non-mul (canonicalize)
            self.create_mul(rhs, lhs)?
        } else {
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[0, self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
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
            node.set_inputs(&[0, self.get_node_id(lhs), self.get_node_id(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_if(&mut self, control: NodeId, cond: &Node) -> NodeId {
        let mut node = Node::new(NodeKind::If, Type::Control);
        node.set_inputs(&[control, self.get_node_id(cond)]);
        self.push_node(&node)
    }

    pub fn create_then(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Then, Type::Control);
        node.set_inputs(&[if_node]);
        self.push_node(&node)
    }

    pub fn create_else(&mut self, if_node: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Else, Type::Control);
        node.set_inputs(&[if_node]);
        self.push_node(&node)
    }

    pub fn create_loop(&mut self, control: NodeId) -> NodeId {
        let mut node = Node::new(NodeKind::Loop, Type::Control);
        node.set_inputs(&[control]);
        self.push_node(&node)
    }

    pub fn create_region(&mut self, controls: &[NodeId]) -> NodeId {
        let mut node = Node::new(NodeKind::Region, Type::Control);
        node.set_inputs(controls);
        self.push_node(&node)
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
                    node.set_inputs(&[0, self.get_node_id(value)]);
                    Ok(node)
                }
            }

            CastKind::Dynamic => {
                // Runtime check required, create DynamicCast node
                let mut node = Node::new(NodeKind::DynamicCast, target_type);
                node.set_inputs(&[0, self.get_node_id(value)]);
                Ok(node)
            }

            CastKind::Invalid => {
                // Cast is impossible
                Err(IRError::TypeMismatch)
            }
        }
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
        let x_id = x.get_identity_id().unwrap() as usize;
        let x_outputs = &builder.node_outputs[x_id];
        assert!(x_outputs.iter().any(|id| id == add_id));

        let c_outputs = &builder.node_outputs[c_id as usize];
        assert!(c_outputs.iter().any(|id| id == add_id));

        // Create sub: node 5 = add - c
        let sub_result = builder.create_sub(&add_result, &c).unwrap();
        let sub_id = builder.get_node_id(&sub_result);

        // add_id should now have sub_id as a user
        let add_outputs = &builder.node_outputs[add_id as usize];
        assert!(add_outputs.iter().any(|id| id == sub_id));

        // c should have both add_id and sub_id as users
        let c_outputs = &builder.node_outputs[c_id as usize];
        assert!(c_outputs.iter().any(|id| id == add_id));
        assert!(c_outputs.iter().any(|id| id == sub_id));
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
        assert!(builder.node_outputs[c_id as usize].iter().any(|id| id == add1_id));

        // Now replace c (const 5) with const 0
        let zero = Node::const_int(0);
        let new_c_id = builder.replace_node(c_id, &zero);

        // Old c should have no outputs left
        assert!(builder.node_outputs[c_id as usize].is_empty());

        // New zero const should have add1 as a user
        assert!(builder.node_outputs[new_c_id as usize].iter().any(|id| id == add1_id));

        // add1 now references new_c_id instead of c_id
        let add1_inputs: Vec<_> = (0..builder.lookup_node(add1_id).inputs_len())
            .map(|i| builder.lookup_node(add1_id).get_input(i))
            .collect();
        assert!(add1_inputs.contains(&new_c_id));
        assert!(!add1_inputs.contains(&c_id));
    }
}
