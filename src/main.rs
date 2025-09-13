mod constraints;
mod types;
use std::collections::HashMap;
use types::*;

pub type NodeId = u32;

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

#[derive(Clone)]
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
            if let Some(interned_id) = self.interned_nodes.get(node) {
                *interned_id
            } else {
                let interned_id = self.nodes.len() as NodeId;
                self.interned_nodes.insert(node.clone(), interned_id);
                interned_id
            }
        }
    }

    pub fn create_add(&mut self, lhs: &Node, rhs: &Node) -> Result<Node, ()> {
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
            let y = self.lookup(r.get_input(1)).clone();
            self.create_sub(lhs, &y)?
        } else if r.kind == NodeKind::Add {
            // x + (y + z) => (x + y) + z
            let y = self.lookup(r.get_input(1)).clone();
            let z = self.lookup(r.get_input(2)).clone();
            let xy = self.create_add(lhs, &y)?;
            self.create_add(&xy, &z)?
        } else {
            let mut node = Node::new(NodeKind::Add, t);
            node.set_inputs(&[self.intern(lhs), self.intern(rhs)]);
            node
        })
    }

    pub fn create_sub(&mut self, _lhs: &Node, _rhs: &Node) -> Result<Node, ()> {
        todo!()
    }

    pub fn create_mul(&mut self, _lhs: &Node, _rhs: &Node) -> Result<Node, ()> {
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
