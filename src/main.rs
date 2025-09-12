use std::{collections::HashMap, num::NonZeroU32};

use ordered_float::OrderedFloat;

pub type NodeId = u32;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoolConstraint {
    Any,
    Const(bool),
}

impl BoolConstraint {
    #[inline]
    pub fn is_const(&self) -> bool {
        return matches!(self, BoolConstraint::Const(..));
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<bool> {
        match self {
            BoolConstraint::Const(value) => Some(*value),
            _ => None,
        }
    }

    #[inline]
    fn combine(self, other: Self, op: impl FnOnce(bool, bool) -> bool) -> Self {
        match (self, other) {
            (BoolConstraint::Const(lval), BoolConstraint::Const(rval)) => {
                BoolConstraint::Const(op(lval, rval))
            }
            _ => BoolConstraint::Any,
        }
    }

    #[inline]
    pub fn and(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval && rval)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct IntConstraint {
    pub min: i64,
    pub max: i64,
}

impl IntConstraint {
    #[inline]
    pub const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn is_const(&self) -> bool {
        self.min == self.max
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<i64> {
        if self.is_const() {
            Some(self.min)
        } else {
            None
        }
    }

    #[inline]
    fn combine_constraints(
        self,
        other: Self,
        lp: IntPrim,
        rp: IntPrim,
        op: impl Fn(Self, Self) -> Option<(i64, i64)>,
    ) -> Result<(IntPrim, Self), ()> {
        // Resolve result primitive type
        let result_prim = match (lp, rp) {
            (l, r) if l == r => l,
            (IntPrim::I64, r) if self.is_const() => r,
            (l, IntPrim::I64) if other.is_const() => l,
            _ => return Err(()),
        };

        let has_const = self.is_const() || other.is_const();

        // Apply operation to get min/max bounds
        let constraint = match op(self, other) {
            Some((min, max)) => {
                let clamped_min = min.max(result_prim.min());
                let clamped_max = max.min(result_prim.max());

                if (min != clamped_min || max != clamped_max) && has_const {
                    return Err(()); // Provable overflow
                }

                Self::new(clamped_min, clamped_max)
            }
            None => {
                if has_const {
                    return Err(()); // Constant overflow
                }
                // Fall back to full range
                Self::new(result_prim.min(), result_prim.max())
            }
        };

        Ok((result_prim, constraint))
    }

    #[inline]
    pub fn add(self, other: Self, lp: IntPrim, rp: IntPrim) -> Result<(IntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((
                a.min.checked_add(b.min)?,
                a.max.checked_add(b.max)?,
            ))
        })
    }

    #[inline]
    pub fn sub(self, other: Self, lp: IntPrim, rp: IntPrim) -> Result<(IntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((
                a.min.checked_sub(b.max)?,
                a.max.checked_sub(b.min)?,
            ))
        })
    }

    #[inline]
    pub fn mul(self, other: Self, lp: IntPrim, rp: IntPrim) -> Result<(IntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            let products = [
                a.min.checked_mul(b.min)?,
                a.min.checked_mul(b.max)?,
                a.max.checked_mul(b.min)?,
                a.max.checked_mul(b.max)?,
            ];
            Some((
                *products.iter().min()?,
                *products.iter().max()?,
            ))
        })
    }

    #[inline]
    pub fn div(self, other: Self, lp: IntPrim, rp: IntPrim) -> Result<(IntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            // Check for division by zero
            if b.min <= 0 && b.max >= 0 {
                return None; // Division by zero possible
            }

            let quotients = [
                a.min.checked_div(b.min)?,
                a.min.checked_div(b.max)?,
                a.max.checked_div(b.min)?,
                a.max.checked_div(b.max)?,
            ];
            Some((
                *quotients.iter().min()?,
                *quotients.iter().max()?,
            ))
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct UIntConstraint {
    pub min: u64,
    pub max: u64,
}

impl UIntConstraint {
    #[inline]
    pub const fn new(min: u64, max: u64) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn is_const(&self) -> bool {
        self.min == self.max
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<u64> {
        if self.is_const() {
            Some(self.min)
        } else {
            None
        }
    }

    #[inline]
    fn combine_constraints(
        self,
        other: Self,
        lp: UIntPrim,
        rp: UIntPrim,
        op: impl Fn(Self, Self) -> Option<(u64, u64)>,
    ) -> Result<(UIntPrim, Self), ()> {
        // Resolve result primitive type
        let result_prim = match (lp, rp) {
            (l, r) if l == r => l,
            (UIntPrim::U64, r) if self.is_const() => r,
            (l, UIntPrim::U64) if other.is_const() => l,
            _ => return Err(()),
        };

        let has_const = self.is_const() || other.is_const();

        // Apply operation to get min/max bounds
        let constraint = match op(self, other) {
            Some((min, max)) => {
                let clamped_min = min.max(result_prim.min());
                let clamped_max = max.min(result_prim.max());

                if (min != clamped_min || max != clamped_max) && has_const {
                    return Err(()); // Provable overflow
                }

                Self::new(clamped_min, clamped_max)
            }
            None => {
                if has_const {
                    return Err(()); // Constant overflow
                }
                // Fall back to full range
                Self::new(result_prim.min(), result_prim.max())
            }
        };

        Ok((result_prim, constraint))
    }

    #[inline]
    pub fn add(self, other: Self, lp: UIntPrim, rp: UIntPrim) -> Result<(UIntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((
                a.min.checked_add(b.min)?,
                a.max.checked_add(b.max)?,
            ))
        })
    }

    #[inline]
    pub fn sub(self, other: Self, lp: UIntPrim, rp: UIntPrim) -> Result<(UIntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((
                a.min.checked_sub(b.max)?,
                a.max.checked_sub(b.min)?,
            ))
        })
    }

    #[inline]
    pub fn mul(self, other: Self, lp: UIntPrim, rp: UIntPrim) -> Result<(UIntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            let products = [
                a.min.checked_mul(b.min)?,
                a.min.checked_mul(b.max)?,
                a.max.checked_mul(b.min)?,
                a.max.checked_mul(b.max)?,
            ];
            Some((
                *products.iter().min()?,
                *products.iter().max()?,
            ))
        })
    }

    #[inline]
    pub fn div(self, other: Self, lp: UIntPrim, rp: UIntPrim) -> Result<(UIntPrim, Self), ()> {
        self.combine_constraints(other, lp, rp, |a, b| {
            // Check for division by zero
            if b.min == 0 {
                return None; // Division by zero possible
            }

            let quotients = [
                a.min.checked_div(b.min)?,
                a.min.checked_div(b.max)?,
                a.max.checked_div(b.min)?,
                a.max.checked_div(b.max)?,
            ];
            Some((
                *quotients.iter().min()?,
                *quotients.iter().max()?,
            ))
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum FloatConstraint {
    Any32,
    Any64,
    Const(OrderedFloat<f64>),
}

impl FloatConstraint {
    #[inline]
    pub fn is_const(&self) -> bool {
        return matches!(self, FloatConstraint::Const(..));
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<f64> {
        match self {
            FloatConstraint::Const(OrderedFloat(value)) => Some(*value),
            _ => None,
        }
    }

    #[inline]
    fn combine(self, other: Self, op: impl FnOnce(f64, f64) -> f64) -> Self {
        match (self, other) {
            (
                FloatConstraint::Const(OrderedFloat(lval)),
                FloatConstraint::Const(OrderedFloat(rval)),
            ) => FloatConstraint::Const(OrderedFloat(op(lval, rval))),
            (
                FloatConstraint::Any32 | FloatConstraint::Const(..),
                FloatConstraint::Any32 | FloatConstraint::Const(..),
            ) => FloatConstraint::Any32,
            (
                FloatConstraint::Any64 | FloatConstraint::Const(..),
                FloatConstraint::Any64 | FloatConstraint::Const(..),
            ) => FloatConstraint::Any64,
            _ => unreachable!("bad type combination"),
        }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval + rval)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum IntPrim {
    I8,
    I16,
    I32,
    I64,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum UIntPrim {
    U8,
    U16,
    U32,
    U64,
}

impl IntPrim {
    pub const fn min(self) -> i64 {
        match self {
            IntPrim::I8 => i8::MIN as i64,
            IntPrim::I16 => i16::MIN as i64,
            IntPrim::I32 => i32::MIN as i64,
            IntPrim::I64 => i64::MIN,
        }
    }

    pub const fn max(self) -> i64 {
        match self {
            IntPrim::I8 => i8::MAX as i64,
            IntPrim::I16 => i16::MAX as i64,
            IntPrim::I32 => i32::MAX as i64,
            IntPrim::I64 => i64::MAX,
        }
    }
}

impl UIntPrim {
    pub const fn min(self) -> u64 {
        match self {
            UIntPrim::U8 => u8::MIN as u64,
            UIntPrim::U16 => u16::MIN as u64,
            UIntPrim::U32 => u32::MIN as u64,
            UIntPrim::U64 => u64::MIN,
        }
    }

    pub const fn max(self) -> u64 {
        match self {
            UIntPrim::U8 => u8::MAX as u64,
            UIntPrim::U16 => u16::MAX as u64,
            UIntPrim::U32 => u32::MAX as u64,
            UIntPrim::U64 => u64::MAX,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Any,   // top
    Never, // bottom

    Control,
    Memory,

    // types for values
    Unit,
    Bool(BoolConstraint),
    Int(IntPrim, IntConstraint),
    UInt(UIntPrim, UIntConstraint),
    Float(FloatConstraint),
}

impl Type {
    pub const UNIT: Type = Type::Unit;
    pub const BOOL: Type = Type::Bool(BoolConstraint::Any);
    pub const I8: Type = Type::prim_int(IntPrim::I8);
    pub const I16: Type = Type::prim_int(IntPrim::I16);
    pub const I32: Type = Type::prim_int(IntPrim::I32);
    pub const I64: Type = Type::prim_int(IntPrim::I64);
    pub const U8: Type = Type::prim_uint(UIntPrim::U8);
    pub const U16: Type = Type::prim_uint(UIntPrim::U16);
    pub const U32: Type = Type::prim_uint(UIntPrim::U32);
    pub const U64: Type = Type::prim_uint(UIntPrim::U64);
    pub const F32: Type = Type::Float(FloatConstraint::Any32);
    pub const F64: Type = Type::Float(FloatConstraint::Any64);

    #[inline]
    pub fn const_bool(value: bool) -> Type {
        Type::Bool(BoolConstraint::Const(value))
    }
    #[inline]
    pub fn const_int(value: i64) -> Type {
        Type::Int(IntPrim::I64, IntConstraint::new(value, value))
    }
    #[inline]
    pub fn const_uint(value: u64) -> Type {
        Type::UInt(UIntPrim::U64, UIntConstraint::new(value, value))
    }
    #[inline]
    pub fn const_float(value: f64) -> Type {
        Type::Float(FloatConstraint::Const(OrderedFloat(value)))
    }

    #[inline]
    pub const fn new_int(t: (IntPrim, IntConstraint)) -> Self {
        Type::Int(t.0, t.1)
    }
    #[inline]
    pub const fn new_uint(t: (UIntPrim, UIntConstraint)) -> Self {
        Type::UInt(t.0, t.1)
    }
    #[inline]
    pub const fn prim_int(prim: IntPrim) -> Self {
        Type::Int(prim, IntConstraint::new(prim.min(), prim.max()))
    }
    #[inline]
    pub const fn prim_uint(prim: UIntPrim) -> Self {
        Type::UInt(prim, UIntConstraint::new(prim.min(), prim.max()))
    }

    #[inline]
    pub fn num_of_same_type(&self, value: u32) -> Node {
        match self {
            Type::Int(..) => Node::const_int(value as i64),
            Type::UInt(..) => Node::const_uint(value as u64),
            Type::Float(..) => Node::const_float(value as f64),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn is_const(&self) -> bool {
        match self {
            Type::Unit => true,
            Type::Bool(c) => c.is_const(),
            Type::Int(_, c) => c.is_const(),
            Type::UInt(_, c) => c.is_const(),
            Type::Float(c) => c.is_const(),
            _ => false,
        }
    }

    #[inline]
    pub fn is_const_zero(&self) -> bool {
        match self {
            Self::Int(_, c) => c.get_const_value() == Some(0),
            Self::UInt(_, c) => c.get_const_value() == Some(0),
            Self::Float(c) => c.get_const_value() == Some(0.0),
            _ => false,
        }
    }

    #[inline]
    pub fn is_const_one(&self) -> bool {
        match self {
            Self::Int(_, c) => c.get_const_value() == Some(1),
            Self::UInt(_, c) => c.get_const_value() == Some(1),
            Self::Float(c) => c.get_const_value() == Some(1.0),
            _ => false,
        }
    }

    #[inline]
    pub fn get_const_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(c) => c.get_const_value(),
            _ => None,
        }
    }

    #[inline]
    pub fn get_const_int(&self) -> Option<i64> {
        match self {
            Self::Int(_, c) => c.get_const_value(),
            _ => None,
        }
    }

    #[inline]
    pub fn get_const_uint(&self) -> Option<u64> {
        match self {
            Self::UInt(_, c) => c.get_const_value(),
            _ => None,
        }
    }

    #[inline]
    pub fn get_const_float(&self) -> Option<f64> {
        match self {
            Self::Float(c) => c.get_const_value(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(NonZeroU32);

#[derive(Default, Copy, Clone, PartialEq, Eq)]
struct NodeListEntry {
    nodes: [NodeId; 3],
    next: u32,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
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
}

#[derive(Copy, Clone)]
union NodeData {
    // only access inputs if inputs_count > 0. if inputs_count > 4 then inputs[3] is a link index for chaining more inputs
    inputs: [NodeId; 4],

    param_index: usize,

    interned_id: NodeId,
}

#[derive(Copy, Clone)]
pub struct Node {
    kind: NodeKind,
    flags: u8,
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
        self.data.inputs = inputs.try_into().unwrap();
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
            *self.interned_nodes.entry(*node).or_insert_with(|| {
                let id = self.nodes.len() as NodeId;
                self.nodes.push(*node);
                id
            })
        }
    }

    pub fn create_add(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, ()> {
        let l = self.resolve(lhs);
        let r = self.resolve(rhs);

        let t = match (l.t, r.t) {
            (Type::Int(lp, lc), Type::Int(rp, rc)) => Type::new_int(lc.add(rc, lp, rp)?),
            (Type::UInt(lp, lc), Type::UInt(rp, rc)) => Type::new_uint(lc.add(rc, lp, rp)?),
            (Type::Float(lc), Type::Float(rc)) => Type::Float(lc.add(rc)),
            _ => unreachable!("bad type combination"),
        };

        Ok(if t.is_const() {
            // constant-folded
            Node::new(NodeKind::Const, t)
        } else if l.t.is_const_zero() {
            // 0 + x => x
            *r
        } else if r.t.is_const_zero() {
            // x + 0 => x
            *l
        } else if l == r {
            // x + x => x * 2
            self.create_mul(lhs, &t.num_of_same_type(2))?
        } else if l.kind != NodeKind::Add && r.kind == NodeKind::Add {
            // non-add + add => add + non-add
            self.create_add(rhs, lhs)?
        } else if r.kind == NodeKind::Neg {
            // x + (-y) => x - y
            let y = *self.lookup(r.get_input(1));
            self.create_sub(lhs, &y)?
        } else if r.kind == NodeKind::Add {
            // x + (y + z) => (x + y) + z
            let y = *self.lookup(r.get_input(1));
            let z = *self.lookup(r.get_input(2));
            let xy = self.create_add(lhs, &y)?;
            self.create_add(&xy, &z)?
        } else {
            let mut node = Node::new(NodeKind::Add, t);
            node.set_inputs(&[self.intern(lhs), self.intern(rhs)]);
            node
        })
    }

    pub fn create_sub(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, ()> {
        todo!()
    }

    pub fn create_mul(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, ()> {
        todo!()
    }

    pub fn create_if(&mut self, cond: &Node) -> Node {
        let mut node = Node::new(NodeKind::If, Type::Control);
        node.set_inputs(&[self.current_control, self.intern(cond)]);
        node
    }
}

fn main() {
    println!("Hello, world!");
}
