mod constraints;
mod types;
use std::collections::HashMap;
use types::*;

pub type NodeId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IRError {
    TypeMismatch,
    IntegerOverflow,
    DivisionByZero,
    InvalidPrimitiveCoercion,
}

#[derive(Default, Copy, Clone, PartialEq, Eq, Debug)]
struct NodeListEntry {
    nodes: [NodeId; 3],
    next: u32,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Unreachable,

    // fake node which contains the ID of an interned node (in data.interned_id).
    // lets us use the Node value based interface even for nodes with identity
    // inputs: none
    Interned,

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

#[derive(Copy, Clone)]
union NodeData {
    // only access inputs if inputs_count > 0. if inputs_count > 4 then inputs[3] is a link index for chaining more inputs
    inputs: [NodeId; 4],

    param_index: usize,

    interned_id: NodeId,
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
    inputs_count: u16,
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
    pub fn create_param(index: usize, t: Type) -> Self {
        let mut node = Self::new(NodeKind::Param, t);
        node.data.param_index = index;
        node
    }

    #[inline]
    pub fn get_input(&self, index: usize) -> NodeId {
        assert!(index < self.inputs_count as usize);
        assert!(index < 4 as usize);
        unsafe { self.data.inputs[index] }
    }

    #[inline]
    pub fn set_inputs(&mut self, inputs: &[NodeId]) {
        assert!(inputs.len() <= 4, "Too many inputs");
        self.inputs_count = inputs.len() as u16;
        unsafe {
            for (i, &input) in inputs.iter().enumerate() {
                self.data.inputs[i] = input;
            }
            // Zero out unused slots
            for i in inputs.len()..4 {
                self.data.inputs[i] = 0;
            }
        }
    }

    #[inline]
    pub fn get_interned_id(&self) -> Option<NodeId> {
        if self.kind == NodeKind::Interned {
            Some(unsafe { self.data.interned_id })
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
        self.inputs_count.hash(state);
        unsafe {
            // inputs takes up the whole data union so we can just hash them no matter the node kind
            self.data.inputs.hash(state);
        }
    }
}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.inputs_count == other.inputs_count
            && unsafe { self.data.inputs == other.data.inputs }
    }
}

impl Eq for Node {}

pub struct IRBuilder {
    nodes: Vec<Node>,
    node_outputs: Vec<NodeListEntry>,
    list_entries: Vec<NodeListEntry>,
    interned_nodes: HashMap<Node, NodeId>,
    interned_list_entries: HashMap<NodeListEntry, usize>,
    current_control: NodeId,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            nodes: vec![
                Node::new(NodeKind::Unreachable, Type::Never),
                Node::new(NodeKind::Entry, Type::Control),
            ],
            node_outputs: vec![NodeListEntry::default(), NodeListEntry::default()],
            list_entries: Vec::new(),
            interned_nodes: HashMap::new(),
            interned_list_entries: HashMap::new(),
            current_control: 1, // start node
        }
    }

    #[inline]
    fn lookup(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline]
    fn resolve<'a>(&'a self, node: &'a Node) -> &Node {
        if let Some(interned_id) = node.get_interned_id() {
            &self.nodes[interned_id as usize]
        } else {
            // TODO: check if we have a refinement node registered for this node (will use to let conditionals refine values in the branches)
            node
        }
    }

    #[inline]
    fn intern(&mut self, node: &Node) -> NodeId {
        if let Some(interned_id) = node.get_interned_id() {
            interned_id
        } else {
            if let Some(interned_id) = self.interned_nodes.get(node) {
                *interned_id
            } else {
                let interned_id = self.nodes.len() as NodeId;
                self.nodes.push(node.clone());
                self.interned_nodes.insert(node.clone(), interned_id);
                interned_id
            }
        }
    }

    pub fn create_add(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve(lhs);
        let r = self.resolve(rhs);

        let t = l.t.add(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            // 0 + x => x
            r.clone()
        } else if r.t.is_const_zero() {
            // x + 0 => x
            l.clone()
        } else if l == r {
            // x + x => x * 2
            self.create_mul(lhs, &Node::num_of_type(2, &t))?
        } else if l.kind != NodeKind::Add && r.kind == NodeKind::Add {
            // non-add + add => add + non-add
            self.create_add(rhs, lhs)?
        } else if r.kind == NodeKind::Neg {
            // x + (-y) => x - y
            let y = self.lookup(r.get_input(1)).clone(); // Neg has input at index 1
            self.create_sub(lhs, &y)?
        } else if r.kind == NodeKind::Add {
            // x + (y + z) => (x + y) + z
            let y = self.lookup(r.get_input(1)).clone(); // Left operand at index 1
            let z = self.lookup(r.get_input(2)).clone(); // Right operand at index 2
            let xy = self.create_add(lhs, &y)?;
            self.create_add(&xy, &z)?
        } else {
            let mut node = Node::new(NodeKind::Add, t);
            node.set_inputs(&[self.intern(lhs), self.intern(rhs)]);
            node
        })
    }

    pub fn create_sub(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve(lhs);
        let r = self.resolve(rhs);

        let t = l.t.sub(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if r.t.is_const_zero() {
            // x - 0 => x
            l.clone()
        } else if l == r {
            // x - x => 0
            Node::num_of_type(0, &t)
        } else if r.kind == NodeKind::Neg {
            // x - (-y) => x + y
            let y = self.lookup(r.get_input(1)).clone();
            self.create_add(lhs, &y)?
        } else if r.kind == NodeKind::Sub {
            // x - (y - z) => x - y + z => (x + z) - y
            let y = self.lookup(r.get_input(1)).clone(); // Left operand at index 1
            let z = self.lookup(r.get_input(2)).clone(); // Right operand at index 2
            let xz = self.create_add(lhs, &z)?;
            self.create_sub(&xz, &y)?
        } else {
            let mut node = Node::new(NodeKind::Sub, t);
            node.set_inputs(&[0, self.intern(lhs), self.intern(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_mul(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve(lhs);
        let r = self.resolve(rhs);

        let t = l.t.mul(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() || r.t.is_const_zero() {
            // x * 0 => 0 or 0 * x => 0
            Node::num_of_type(0, &t)
        } else if l.t.is_const_one() {
            // 1 * x => x
            r.clone()
        } else if r.t.is_const_one() {
            // x * 1 => x
            l.clone()
        } else if l == r {
            // x * x => x^2 (no special peephole for now)
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[self.intern(lhs), self.intern(rhs)]);
            node
        } else if l.kind != NodeKind::Mul && r.kind == NodeKind::Mul {
            // non-mul * mul => mul * non-mul (canonicalize)
            self.create_mul(rhs, lhs)?
        } else {
            let mut node = Node::new(NodeKind::Mul, t);
            node.set_inputs(&[0, self.intern(lhs), self.intern(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_div(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, IRError> {
        let l = self.resolve(lhs);
        let r = self.resolve(rhs);

        let t = l.t.div(&r.t)?;

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            // 0 / x => 0 (assuming x != 0, which is checked by t.div())
            Node::num_of_type(0, &t)
        } else if r.t.is_const_one() {
            // x / 1 => x
            l.clone()
        } else if l == r {
            // x / x => 1 (assuming x != 0, which is checked by t.div())
            Node::num_of_type(1, &t)
        } else {
            let mut node = Node::new(NodeKind::Div, t);
            node.set_inputs(&[0, self.intern(lhs), self.intern(rhs)]); // 0 = no control dependency
            node
        })
    }

    pub fn create_if(&mut self, cond: &Node) -> Node {
        let mut node = Node::new(NodeKind::If, Type::Control);
        node.set_inputs(&[self.current_control, self.intern(cond)]);
        node
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
                    node.set_inputs(&[self.current_control, self.intern(value)]);
                    Ok(node)
                }
            }
            
            CastKind::Dynamic => {
                // Runtime check required, create DynamicCast node
                let mut node = Node::new(NodeKind::DynamicCast, target_type);
                node.set_inputs(&[self.current_control, self.intern(value)]);
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

        let x = Node::create_param(0, Type::I32);
        let zero = Node::const_int(0);
        let one = Node::const_int(1);

        // Test x + 0 => x
        let result = builder.create_add(&x, &zero).unwrap();
        assert_eq!(result, x);

        // Test 0 + x => x
        let result = builder.create_add(&zero, &x).unwrap();
        assert_eq!(result, x);

        // Test x - 0 => x
        let result = builder.create_sub(&x, &zero).unwrap();
        assert_eq!(result, x);

        // Test x - x => 0
        let result = builder.create_sub(&x, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(0));

        // Test x * 1 => x
        let result = builder.create_mul(&x, &one).unwrap();
        assert_eq!(result, x);

        // Test 1 * x => x
        let result = builder.create_mul(&one, &x).unwrap();
        assert_eq!(result, x);

        // Test x * 0 => 0
        let result = builder.create_mul(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_int(), Some(0));

        // Test x / 1 => x
        let result = builder.create_div(&x, &one).unwrap();
        assert_eq!(result, x);

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

        let x = Node::create_param(0, Type::I32);
        let two = Node::const_int(2);

        // Test x + x => x * 2 (from create_add)
        let result = builder.create_add(&x, &x).unwrap();
        assert_eq!(result.kind, NodeKind::Mul);
        // Should be x * 2
        let mul_rhs = builder.lookup(result.get_input(2)); // Right operand is at index 2
        assert_eq!(mul_rhs.t.get_const_int(), Some(2));

        // Test double negation: x - (-y) => x + y
        let mut neg_x = Node::new(NodeKind::Neg, Type::I32);
        neg_x.set_inputs(&[0, builder.intern(&x)]); // 0 = no control, x as operand
        let _result = builder.create_sub(&two, &neg_x);
        // This should recursively call create_add, but we can't easily test without more setup
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
        let x = Node::create_param(0, Type::F64);
        let zero = Node::const_float(0.0);
        let one = Node::const_float(1.0);

        // x * 0.0 => 0.0
        let result = builder.create_mul(&x, &zero).unwrap();
        assert_eq!(result.kind, NodeKind::Const);
        assert_eq!(result.t.get_const_float(), Some(0.0));

        // x / 1.0 => x
        let result = builder.create_div(&x, &one).unwrap();
        assert_eq!(result, x);
    }

    #[test]
    fn test_mixed_type_arithmetic_errors() {
        let int_node = Node::const_int(5);
        let float_node = Node::const_float(5.0);
        let bool_node = Node::const_bool(true);

        // Test that mixed types properly error
        let int_type = int_node.t;
        let float_type = float_node.t;
        let bool_type = bool_node.t;

        // int + float should error
        assert!(int_type.add(&float_type).is_err());

        // int + bool should error
        assert!(int_type.add(&bool_type).is_err());

        // float + bool should error
        assert!(float_type.add(&bool_type).is_err());
    }

    #[test]
    fn test_cast_functionality() {
        use crate::types::CastKind;
        use crate::constraints::{IntConstraint, UIntConstraint};
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
        assert_eq!(signed_positive.cast_kind(&unsigned_compatible), CastKind::Static);

        let signed_negative = Type::Int(IntPrim::I32, IntConstraint::new(-50, 10));
        let unsigned_range = Type::UInt(UIntPrim::U32, UIntConstraint::new(0, 100));
        assert_eq!(signed_negative.cast_kind(&unsigned_range), CastKind::Dynamic);

        // Test bool casts
        let bool_true = Type::Bool(BoolConstraint::Const(true));
        let bool_false = Type::Bool(BoolConstraint::Const(false));
        let bool_any = Type::Bool(BoolConstraint::Any);

        assert_eq!(bool_true.cast_kind(&bool_any), CastKind::Static);   // widening
        assert_eq!(bool_any.cast_kind(&bool_true), CastKind::Dynamic);  // narrowing
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
}
