use std::sync::Arc;

use ordered_float::OrderedFloat;

use crate::IRError;
use crate::constraints::*;
use crate::symbols::SymbolId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    Static,  // Guaranteed safe, no runtime check needed
    Dynamic, // Possible but requires runtime type check
    Invalid, // Definitely impossible
}

// Private zero-sized type to prevent direct construction of Union/Error
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Priv {
    _private: (),
}

impl Priv {
    fn new() -> Self {
        Priv { _private: () }
    }
}

#[derive(Debug, Copy, Clone)]
enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DataField {
    pub name: SymbolId, // Field name (e.g. "x", "y" for records, "0", "1" for tuples)
    pub ty: Type,       // Field type
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    Any,   // top
    Never, // bottom

    Control, // type of pure control nodes (other side effecting nodes like Call which are typed with their results can still be in control chain)
    Memory,  // type of memory nodes (except Load which has the type of the loaded value)

    // types for values
    Unit,
    Bool(BoolConstraint),
    Int(IntPrim, IntConstraint),
    UInt(UIntPrim, UIntConstraint),
    Float(FloatConstraint),
    Type(TypeConstraint),

    Data(Option<SymbolId>, Arc<[DataField]>), // tag, fields

    Union(Arc<[Type]>, Priv),

    Error(Arc<Type>, Priv),
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
    pub const F32: Type = Type::Float(FloatConstraint::Any(FloatPrim::F32));
    pub const F64: Type = Type::Float(FloatConstraint::Any(FloatPrim::F64));

    /// Generic helper for binary arithmetic operations
    fn apply_arithmetic_op(&self, other: &Self, op: ArithmeticOp) -> Result<Self, IRError> {
        match (self, other) {
            (Type::Int(lp, lc), Type::Int(rp, rc)) => {
                let result = match op {
                    ArithmeticOp::Add => lc.add(*rc, *lp, *rp)?,
                    ArithmeticOp::Sub => lc.sub(*rc, *lp, *rp)?,
                    ArithmeticOp::Mul => lc.mul(*rc, *lp, *rp)?,
                    ArithmeticOp::Div => lc.div(*rc, *lp, *rp)?,
                };
                Ok(Type::new_int(result))
            }
            (Type::UInt(lp, lc), Type::UInt(rp, rc)) => {
                let result = match op {
                    ArithmeticOp::Add => lc.add(*rc, *lp, *rp)?,
                    ArithmeticOp::Sub => lc.sub(*rc, *lp, *rp)?,
                    ArithmeticOp::Mul => lc.mul(*rc, *lp, *rp)?,
                    ArithmeticOp::Div => lc.div(*rc, *lp, *rp)?,
                };
                Ok(Type::new_uint(result))
            }
            (Type::Float(lc), Type::Float(rc)) => {
                let result = match op {
                    ArithmeticOp::Add => lc.add(*rc),
                    ArithmeticOp::Sub => lc.sub(*rc),
                    ArithmeticOp::Mul => lc.mul(*rc),
                    ArithmeticOp::Div => lc.div(*rc),
                };
                Ok(Type::Float(result))
            }
            (Type::Union(types, ..), other) => {
                let results: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| t.apply_arithmetic_op(other, op))
                    .collect();
                Ok(Self::make_union(results?))
            }
            (other, Type::Union(types, ..)) => {
                let results: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| other.apply_arithmetic_op(t, op))
                    .collect();
                Ok(Self::make_union(results?))
            }
            _ => Err(IRError::TypeMismatch),
        }
    }

    // Helper constructors for Data types
    pub fn make_tuple(types: Vec<Type>) -> Type {
        Self::make_data(
            None,
            types
                .into_iter()
                .enumerate()
                .map(|(i, ty)| (i.to_string(), ty)),
        )
    }

    pub fn make_named_tuple(name: &str, types: Vec<Type>) -> Type {
        Self::make_data(
            Some(name),
            types
                .into_iter()
                .enumerate()
                .map(|(i, ty)| (i.to_string(), ty)),
        )
    }

    pub fn make_record(fields: Vec<(&str, Type)>) -> Type {
        Self::make_data(
            None,
            fields.into_iter().map(|(name, ty)| (name.to_string(), ty)),
        )
    }

    pub fn make_named_record(name: &str, fields: Vec<(&str, Type)>) -> Type {
        Self::make_data(
            Some(name),
            fields.into_iter().map(|(name, ty)| (name.to_string(), ty)),
        )
    }

    fn make_data(tag: Option<&str>, fields: impl Iterator<Item = (String, Type)>) -> Type {
        use crate::symbols::intern_symbol;
        let tag_id = tag.map(|s| intern_symbol(s));
        let data_fields: Arc<[DataField]> = fields
            .map(|(name, ty)| DataField {
                name: intern_symbol(&name),
                ty,
            })
            .collect();
        Type::Data(tag_id, data_fields)
    }

    pub fn make_union(mut types: Vec<Type>) -> Type {
        // Flatten nested unions and distribute errors over unions
        let mut i = 0;
        while i < types.len() {
            match &types[i] {
                Type::Union(..) => {
                    let union_type = std::mem::replace(&mut types[i], Type::Unit);
                    if let Type::Union(inner_types, ..) = union_type {
                        assert!(
                            !inner_types.is_empty(),
                            "unions should never be empty, by construction"
                        );
                        types[i] = inner_types[0].clone();
                        for t in inner_types.iter().skip(1) {
                            types.push(t.clone());
                        }
                    }
                }
                Type::Error(inner, ..) => {
                    // If we have Error(Union([A, B])), distribute to [Error(A), Error(B)]
                    if let Type::Union(..) = &**inner {
                        let error_type = std::mem::replace(&mut types[i], Type::Unit);
                        let Type::Error(inner_content, ..) = error_type else {
                            unreachable!()
                        };
                        let Type::Union(inner_types, ..) =
                            Arc::try_unwrap(inner_content).unwrap_or_else(|arc| (*arc).clone())
                        else {
                            unreachable!()
                        };

                        // Replace current position with Error(first_type)
                        types[i] = Type::Error(Arc::new(inner_types[0].clone()), Priv::new());

                        // Push Error(remaining_types) directly
                        for t in inner_types.iter().skip(1) {
                            types.push(Type::Error(Arc::new(t.clone()), Priv::new()));
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }

        // Sort and remove duplicates - auto-derived Ord does exactly what we want
        types.sort();
        types.dedup();

        // Check if Any is present - it subsumes everything
        if types.iter().any(|t| matches!(t, Type::Any)) {
            return Type::Any;
        }

        // Remove Never types - they add no information to a union
        types.retain(|t| !matches!(t, Type::Never));

        // Merge overlapping/adjacent ranges in a single pass
        if types.len() > 1 {
            let mut write_pos = 0;
            let mut read_pos = 1;

            while read_pos < types.len() {
                // Try to merge types - returns Some(merged) if possible, None if not
                let merged = Self::try_merge_types(&types[write_pos], &types[read_pos]);

                if let Some(merged_type) = merged {
                    types[write_pos] = merged_type;
                    // Continue reading without advancing write_pos
                } else {
                    // Can't merge, advance write position and copy
                    write_pos += 1;
                    if write_pos != read_pos {
                        types[write_pos] = types[read_pos].clone();
                    }
                }
                read_pos += 1;
            }

            types.truncate(write_pos + 1);
        }

        match types.len() {
            0 => Type::Never,
            1 => types.into_iter().next().unwrap(),
            _ => Type::Union(types.into(), Priv::new()),
        }
    }

    /// Try to merge two types in a union. Returns Some(merged) if possible, None if not.
    fn try_merge_types(left: &Type, right: &Type) -> Option<Type> {
        if left == right {
            // trivial merge of identical types
            return Some(left.clone());
        }

        match (left, right) {
            (Type::Bool(lc), Type::Bool(rc)) => {
                // Any Bool constraint subsumes more specific ones
                match (lc, rc) {
                    (BoolConstraint::Any, _) | (_, BoolConstraint::Any) => {
                        Some(Type::Bool(BoolConstraint::Any))
                    }
                    (BoolConstraint::Const(a), BoolConstraint::Const(b)) if a == b => {
                        Some(Type::Bool(*lc))
                    }
                    (BoolConstraint::Const(_), BoolConstraint::Const(_)) => {
                        Some(Type::Bool(BoolConstraint::Any))
                    } // true ∪ false = Any
                }
            }

            (Type::Int(lp, lc), Type::Int(rp, rc)) if lp == rp => {
                // Adjacent or overlapping: [a,b] and [c,d] merge if b+1 >= c
                if lc.max.saturating_add(1) >= rc.min {
                    Some(Type::Int(
                        *lp,
                        IntConstraint::new(lc.min, lc.max.max(rc.max)),
                    ))
                } else {
                    None
                }
            }

            (Type::UInt(lp, lc), Type::UInt(rp, rc)) if lp == rp => {
                // Adjacent or overlapping with overflow protection
                if lc.max.saturating_add(1) >= rc.min {
                    Some(Type::UInt(
                        *lp,
                        UIntConstraint::new(lc.min, lc.max.max(rc.max)),
                    ))
                } else {
                    None
                }
            }

            (Type::Float(FloatConstraint::Any(prim)), Type::Float(FloatConstraint::Const(_)))
            | (Type::Float(FloatConstraint::Const(_)), Type::Float(FloatConstraint::Any(prim))) => {
                // Polymorphic constant gets subsumed by specific primitive
                Some(Type::Float(FloatConstraint::Any(*prim)))
            }

            (Type::Type(TypeConstraint::Any), Type::Type(TypeConstraint::Const(_)))
            | (Type::Type(TypeConstraint::Const(_)), Type::Type(TypeConstraint::Any)) => {
                Some(Type::Type(TypeConstraint::Any))
            }
            _ => None, // Can't merge
        }
    }

    pub fn make_error(inner: Type) -> Type {
        match inner {
            Type::Never => Type::Never,
            Type::Error(..) => inner,
            Type::Union(types, ..) => {
                // Distribute: Error(Union([A, B])) → make_union([Error(A), Error(B)])
                let error_types: Vec<Type> = types
                    .iter()
                    .map(|t| Type::Error(Arc::new(t.clone()), Priv::new()))
                    .collect();
                Type::make_union(error_types) // This will place errors at tail
            }
            other => Type::Error(Arc::new(other), Priv::new()),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, IRError> {
        self.apply_arithmetic_op(other, ArithmeticOp::Add)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, IRError> {
        self.apply_arithmetic_op(other, ArithmeticOp::Sub)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, IRError> {
        self.apply_arithmetic_op(other, ArithmeticOp::Mul)
    }

    pub fn div(&self, other: &Self) -> Result<Self, IRError> {
        self.apply_arithmetic_op(other, ArithmeticOp::Div)
    }

    pub fn intersect(&self, other: &Self) -> Type {
        if self == other {
            return self.clone();
        }

        match (self, other) {
            (Type::Bool(lc), Type::Bool(rc)) => match (lc, rc) {
                (BoolConstraint::Const(a), BoolConstraint::Const(b)) if a == b => self.clone(),
                (BoolConstraint::Const(_), BoolConstraint::Const(_)) => Type::Never,
                (BoolConstraint::Const(_), BoolConstraint::Any) => self.clone(),
                (BoolConstraint::Any, BoolConstraint::Const(_)) => other.clone(),
                (BoolConstraint::Any, BoolConstraint::Any) => self.clone(),
            },

            // Same type → intersection is the narrower constraint
            (Type::Int(lp, lc), Type::Int(rp, rc)) if lp == rp => {
                let min = lc.min.max(rc.min);
                let max = lc.max.min(rc.max);
                if min <= max {
                    Type::Int(*lp, IntConstraint::new(min, max))
                } else {
                    Type::Never // Empty intersection
                }
            }

            (Type::UInt(lp, lc), Type::UInt(rp, rc)) if lp == rp => {
                let min = lc.min.max(rc.min);
                let max = lc.max.min(rc.max);
                if min <= max {
                    Type::UInt(*lp, UIntConstraint::new(min, max))
                } else {
                    Type::Never // Empty intersection
                }
            }

            // Data type intersections - handle struct coercion
            (Type::Data(tag1, fields1), Type::Data(tag2, fields2)) => {
                if fields1 == fields2 {
                    match (tag1, tag2) {
                        // Identical untagged structs
                        (None, None) => self.clone(),
                        // Identical tagged structs
                        (Some(a), Some(b)) if a == b => self.clone(),
                        // Auto-coercion cases
                        (None, Some(_)) => other.clone(), // untagged -> tagged
                        (Some(_), None) => other.clone(), // tagged -> untagged
                        // Different tags with same fields -> incompatible
                        (Some(_), Some(_)) => Type::Never,
                    }
                } else {
                    Type::Never // Different fields
                }
            }

            // Union intersections - delegate to the specialized logic
            (Type::Union(..), other) => other.intersect(self),

            // Union intersection with ambiguity handling
            (source, Type::Union(types, ..)) => {
                // First check for exact match - this has highest priority
                if types.iter().any(|t| source == t) {
                    return source.clone(); // Exact match wins
                }

                let mut tagged_match = None;
                let mut untagged_match = None;
                let mut other_match = None;
                let mut tagged_count = 0;
                let mut untagged_count = 0;
                let mut other_count = 0;

                for t in types.iter() {
                    let intersection = source.intersect(t);
                    if !matches!(intersection, Type::Never) {
                        match intersection {
                            Type::Data(Some(_), _) => {
                                tagged_count += 1;
                                tagged_match = Some(intersection);
                            }
                            Type::Data(None, _) => {
                                untagged_count += 1;
                                untagged_match = Some(intersection);
                            }
                            other => {
                                other_count += 1;
                                other_match = Some(other);
                            }
                        }
                    }
                }

                match (tagged_count, untagged_count, other_count) {
                    (0, 0, 0) => Type::Never,             // No matches
                    (1, 0, 0) => tagged_match.unwrap(),   // Single tagged
                    (0, 1, 0) => untagged_match.unwrap(), // Single untagged
                    (0, 0, 1) => other_match.unwrap(),    // Single other
                    (1, _, 0) => tagged_match.unwrap(),   // Prefer tagged over untagged
                    (_, _, _) => Type::Never,             // Any ambiguity = error
                }
            }

            // Error intersections
            (Type::Error(la, ..), Type::Error(ra, ..)) => {
                let inner_intersection = (**la).clone().intersect(&(**ra));
                match inner_intersection {
                    Type::Never => Type::Never,
                    other => Type::make_error(other),
                }
            }

            // Different types → no intersection
            _ => Type::Never,
        }
    }

    pub fn subtract(&self, other: &Self) -> Type {
        if self == other {
            return Type::Never;
        }

        match (self, other) {
            (Type::Bool(lc), Type::Bool(rc)) => {
                match (lc, rc) {
                    (BoolConstraint::Const(a), BoolConstraint::Const(b)) if a == b => Type::Never,
                    (BoolConstraint::Const(_), BoolConstraint::Const(_)) => self.clone(), // Different constants
                    (BoolConstraint::Any, BoolConstraint::Const(b)) => Type::const_bool(!b),
                    (BoolConstraint::Const(_), BoolConstraint::Any) => Type::Never,
                    (BoolConstraint::Any, BoolConstraint::Any) => Type::Never,
                }
            }

            // Same range type → compute set difference
            (Type::Int(lp, lc), Type::Int(rp, rc)) if lp == rp => {
                // [a,b] - [c,d] can result in [a,c-1] ∪ [d+1,b]
                let mut ranges = Vec::new();

                // Left part: [a, min(b, c-1)]
                if lc.min < rc.min {
                    let left_max = lc.max.min(rc.min - 1);
                    if lc.min <= left_max {
                        ranges.push(Type::Int(*lp, IntConstraint::new(lc.min, left_max)));
                    }
                }

                // Right part: [max(a, d+1), b]
                if lc.max > rc.max {
                    let right_min = lc.min.max(rc.max + 1);
                    if right_min <= lc.max {
                        ranges.push(Type::Int(*lp, IntConstraint::new(right_min, lc.max)));
                    }
                }

                Type::make_union(ranges)
            }

            (Type::UInt(lp, lc), Type::UInt(rp, rc)) if lp == rp => {
                let mut ranges = Vec::new();

                // Left part: [a, min(b, c-1)] with underflow protection
                if lc.min < rc.min {
                    let left_max = lc.max.min(rc.min.saturating_sub(1));
                    if lc.min <= left_max {
                        ranges.push(Type::UInt(*lp, UIntConstraint::new(lc.min, left_max)));
                    }
                }

                // Right part: [max(a, d+1), b] with overflow protection
                if lc.max > rc.max && rc.max < u64::MAX {
                    let right_min = lc.min.max(rc.max + 1);
                    if right_min <= lc.max {
                        ranges.push(Type::UInt(*lp, UIntConstraint::new(right_min, lc.max)));
                    }
                }

                Type::make_union(ranges)
            }

            // Error subtraction
            (Type::Error(la, ..), Type::Error(ra, ..)) => {
                let inner_subtraction = (**la).clone().subtract(&(**ra));
                match inner_subtraction {
                    Type::Never => Type::Never,
                    other => Type::make_error(other),
                }
            }

            // Union subtraction
            (Type::Union(types, ..), other) => {
                let subtracted: Vec<Type> = types
                    .iter()
                    .map(|t| t.subtract(other))
                    .filter(|t| !matches!(t, Type::Never))
                    .collect();
                Type::make_union(subtracted)
            }

            // Different types → self unchanged
            _ => self.clone(),
        }
    }

    #[inline]
    fn bool_from_option(result: Option<bool>) -> Type {
        match result {
            Some(value) => Type::const_bool(value),
            None => Type::BOOL,
        }
    }

    #[inline]
    fn compare(&self, other: &Self, op: CompareOp) -> Result<Type, IRError> {
        match (self, other) {
            (Type::Bool(lc), Type::Bool(rc)) => Ok(Type::bool_from_option(lc.compare(*rc, op))),
            (Type::Int(lp, lc), Type::Int(rp, rc)) if lp == rp => {
                Ok(Type::bool_from_option(lc.compare(*rc, op)))
            }
            (Type::UInt(lp, lc), Type::UInt(rp, rc)) if lp == rp => {
                Ok(Type::bool_from_option(lc.compare(*rc, op)))
            }
            (Type::Float(lc), Type::Float(rc)) => Ok(Type::bool_from_option(lc.compare(*rc, op))),
            (Type::Type(lc), Type::Type(rc)) => Ok(Type::bool_from_option(lc.compare(rc, op))),
            (Type::Union(types, ..), other) => {
                let results: Result<Vec<_>, _> =
                    types.iter().map(|t| t.compare(other, op)).collect();
                Ok(Self::make_union(results?))
            }
            (other, Type::Union(types, ..)) => {
                let results: Result<Vec<_>, _> =
                    types.iter().map(|t| other.compare(t, op)).collect();
                Ok(Self::make_union(results?))
            }
            _ => Err(IRError::TypeMismatch),
        }
    }

    pub fn eq(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::Eq)
    }

    pub fn neq(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::Neq)
    }

    pub fn lt(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::Lt)
    }

    pub fn gt(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::Gt)
    }

    pub fn lteq(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::LtEq)
    }

    pub fn gteq(&self, other: &Self) -> Result<Type, IRError> {
        self.compare(other, CompareOp::GtEq)
    }

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
    pub fn is_const(&self) -> bool {
        match self {
            Type::Unit => true,
            Type::Bool(c) => c.is_const(),
            Type::Int(_, c) => c.is_const(),
            Type::UInt(_, c) => c.is_const(),
            Type::Float(c) => c.is_const(),
            Type::Type(c) => c.is_const(),
            Type::Data(..) => false, // Data types are not primitive constants
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

    #[inline]
    pub fn get_const_type(&self) -> Option<Arc<Type>> {
        match self {
            Self::Type(c) => c.get_const_value(),
            _ => None,
        }
    }

    /// Helper function to determine cast kind based on range relationships
    fn range_cast_kind(from_range: CommonRange, to_range: CommonRange) -> CastKind {
        if to_range.contains(&from_range) {
            CastKind::Static // Source range ⊆ target range
        } else if from_range.is_disjoint(&to_range) {
            CastKind::Invalid // Ranges are disjoint
        } else {
            CastKind::Dynamic // Ranges overlap but source not contained in target
        }
    }

    /// Determine what kind of cast is needed from self to target_type
    pub fn cast_kind(&self, target_type: &Type) -> CastKind {
        // Identity cast - always static
        if self == target_type {
            return CastKind::Static;
        }

        match (self, target_type) {
            // Bool casts - write out all cases
            (Type::Bool(BoolConstraint::Const(a)), Type::Bool(BoolConstraint::Const(b))) => {
                if a == b {
                    CastKind::Static // Same constant
                } else {
                    CastKind::Invalid // Different constants (true ≠ false)
                }
            }
            (Type::Bool(BoolConstraint::Const(_)), Type::Bool(BoolConstraint::Any)) => {
                CastKind::Static // Const to Any is widening
            }
            (Type::Bool(BoolConstraint::Any), Type::Bool(BoolConstraint::Const(_))) => {
                CastKind::Dynamic // Any to Const needs runtime check
            }
            (Type::Bool(BoolConstraint::Any), Type::Bool(BoolConstraint::Any)) => {
                CastKind::Static // Any to Any (already handled by identity check above)
            }

            // Signed to signed
            (Type::Int(_, from_constraint), Type::Int(_, to_constraint)) => Self::range_cast_kind(
                CommonRange::from(*from_constraint),
                CommonRange::from(*to_constraint),
            ),

            // Unsigned to unsigned
            (Type::UInt(_, from_constraint), Type::UInt(_, to_constraint)) => {
                Self::range_cast_kind(
                    CommonRange::from(*from_constraint),
                    CommonRange::from(*to_constraint),
                )
            }

            // Signed to unsigned
            (Type::Int(_, from_constraint), Type::UInt(_, to_constraint)) => Self::range_cast_kind(
                CommonRange::from(*from_constraint),
                CommonRange::from(*to_constraint),
            ),

            // Unsigned to signed
            (Type::UInt(_, from_constraint), Type::Int(_, to_constraint)) => Self::range_cast_kind(
                CommonRange::from(*from_constraint),
                CommonRange::from(*to_constraint),
            ),

            // Other cross-type casts (int/uint to float, etc.)
            (Type::Int(..) | Type::UInt(..), Type::Float(..)) => CastKind::Dynamic,
            (Type::Float(..), Type::Int(..) | Type::UInt(..)) => CastKind::Dynamic,

            // Type casts - write out all cases
            (Type::Type(TypeConstraint::Const(a)), Type::Type(TypeConstraint::Const(b))) => {
                if a == b {
                    CastKind::Static // Same constant
                } else {
                    CastKind::Invalid // Different constants (true ≠ false)
                }
            }
            (Type::Type(TypeConstraint::Const(_)), Type::Type(TypeConstraint::Any)) => {
                CastKind::Static // Const to Any is widening
            }
            (Type::Type(TypeConstraint::Any), Type::Type(TypeConstraint::Const(_))) => {
                CastKind::Dynamic // Any to Const needs runtime check
            }
            (Type::Type(TypeConstraint::Any), Type::Type(TypeConstraint::Any)) => {
                CastKind::Static // Any to Any (already handled by identity check above)
            }

            // Union casts - check all possibilities
            (Type::Union(types, ..), target) => {
                let cast_kinds: Vec<_> = types.iter().map(|t| t.cast_kind(target)).collect();

                if cast_kinds.iter().all(|&k| k == CastKind::Static) {
                    CastKind::Static // All members cast statically
                } else if cast_kinds.iter().any(|&k| k == CastKind::Invalid) {
                    CastKind::Invalid // Some members can't cast at all
                } else {
                    CastKind::Dynamic // Mix of static and dynamic, or all dynamic
                }
            }

            // Cast to union - use lattice algebra
            (source, Type::Union(..)) => {
                let intersection = source.intersect(target_type);
                if intersection == Type::Never {
                    CastKind::Invalid // No compatible types
                } else if intersection == *source {
                    CastKind::Static // Source is subset of union
                } else {
                    CastKind::Dynamic // Partial compatibility
                }
            }

            // Special types
            (Type::Error(..), _) | (_, Type::Error(..)) => CastKind::Invalid,
            (Type::Control, _) | (_, Type::Control) => CastKind::Invalid,
            (Type::Memory, _) | (_, Type::Memory) => CastKind::Invalid,
            (Type::Never, _) => CastKind::Static, // Never can cast to anything (vacuously)
            (_, Type::Never) => CastKind::Invalid, // Nothing can cast to Never
            (Type::Any, _) => CastKind::Dynamic,  // Any might be anything at runtime
            (_, Type::Any) => CastKind::Static,   // Everything can upcast to Any

            // Data types - use intersection to check coercion compatibility
            (Type::Data(..), Type::Data(..)) => {
                let intersection = self.intersect(target_type);
                if intersection == *target_type {
                    CastKind::Static // Successful coercion
                } else {
                    CastKind::Invalid // No coercion possible
                }
            }
            (Type::Data(..), _) | (_, Type::Data(..)) => CastKind::Invalid,

            // Everything else is invalid
            _ => CastKind::Invalid,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeConstraint {
    Any,
    Const(Arc<Type>),
}

impl TypeConstraint {
    #[inline]
    pub fn is_const(&self) -> bool {
        return matches!(self, TypeConstraint::Const(..));
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<Arc<Type>> {
        match self {
            TypeConstraint::Const(t) => Some(t.clone()),
            _ => None,
        }
    }

    pub fn compare(&self, other: &Self, op: CompareOp) -> Option<bool> {
        match (self, other) {
            (TypeConstraint::Const(a), TypeConstraint::Const(b)) => Some(match op {
                CompareOp::Lt => a < b,
                CompareOp::Gt => a > b,
                CompareOp::LtEq => a <= b,
                CompareOp::GtEq => a >= b,
                CompareOp::Eq => a == b,
                CompareOp::Neq => a != b,
            }),
            _ => None, // Conservative for non-constant types
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_union_basic() {
        // Single type should return that type
        let result = Type::make_union(vec![Type::I32]);
        assert_eq!(result, Type::I32);

        // Empty union should return Never
        let result = Type::make_union(vec![]);
        assert_eq!(result, Type::Never);
    }

    #[test]
    fn test_make_union_any_subsumes() {
        // Any should subsume everything
        let result = Type::make_union(vec![Type::I32, Type::Any, Type::Bool(BoolConstraint::Any)]);
        assert_eq!(result, Type::Any);
    }

    #[test]
    fn test_make_union_never_filtered() {
        // Never should be filtered out of unions
        let result = Type::make_union(vec![
            Type::I32,
            Type::Never,
            Type::Bool(BoolConstraint::Any),
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                assert!(!types.iter().any(|t| matches!(t, Type::Never)));
            }
            _ => panic!("Expected union"),
        }

        // All Never should return Never
        let result = Type::make_union(vec![Type::Never, Type::Never]);
        assert_eq!(result, Type::Never);
    }

    #[test]
    fn test_make_union_range_merging() {
        // Overlapping ranges should merge
        let result = Type::make_union(vec![
            Type::Int(IntPrim::I32, IntConstraint::new(1, 5)),
            Type::Int(IntPrim::I32, IntConstraint::new(4, 8)),
        ]);
        assert_eq!(result, Type::Int(IntPrim::I32, IntConstraint::new(1, 8)));

        // Adjacent ranges should merge
        let result = Type::make_union(vec![
            Type::Int(IntPrim::I32, IntConstraint::new(1, 5)),
            Type::Int(IntPrim::I32, IntConstraint::new(6, 10)),
        ]);
        assert_eq!(result, Type::Int(IntPrim::I32, IntConstraint::new(1, 10)));

        // Disjoint ranges should remain separate
        let result = Type::make_union(vec![
            Type::Int(IntPrim::I32, IntConstraint::new(1, 3)),
            Type::Int(IntPrim::I32, IntConstraint::new(7, 9)),
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                assert_eq!(types[0], Type::Int(IntPrim::I32, IntConstraint::new(1, 3)));
                assert_eq!(types[1], Type::Int(IntPrim::I32, IntConstraint::new(7, 9)));
            }
            _ => panic!("Expected union of disjoint ranges"),
        }
    }

    #[test]
    fn test_make_union_nested_flattening() {
        // Nested unions should be flattened
        let inner_union = Type::make_union(vec![Type::I32, Type::Bool(BoolConstraint::Any)]);

        let result = Type::make_union(vec![Type::I64, inner_union, Type::F32]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 4);
                // Should contain all flattened types, no nested unions
                assert!(!types.iter().any(|t| matches!(t, Type::Union(..))));
            }
            _ => panic!("Expected flattened union"),
        }
    }

    #[test]
    fn test_make_union_deduplication() {
        // Duplicate types should be removed
        let result = Type::make_union(vec![
            Type::I32,
            Type::I32,
            Type::Bool(BoolConstraint::Any),
            Type::I32,
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                // Should be sorted: Bool < Int
                assert!(matches!(types[0], Type::Bool(_)));
                assert_eq!(types[1], Type::I32);
            }
            _ => panic!("Expected deduplicated union"),
        }
    }

    #[test]
    fn test_make_union_complex_case() {
        // Complex case: nested unions + Never + overlapping ranges + Any
        let nested = Type::make_union(vec![
            Type::Int(IntPrim::I32, IntConstraint::new(10, 15)),
            Type::Never,
        ]);

        let result = Type::make_union(vec![
            Type::Int(IntPrim::I32, IntConstraint::new(1, 5)),
            nested,
            Type::Int(IntPrim::I32, IntConstraint::new(12, 20)), // Overlaps with nested
            Type::Never,
            Type::Any,
            Type::Bool(BoolConstraint::Const(true)),
        ]);

        // Any should subsume everything
        assert_eq!(result, Type::Any);
    }

    #[test]
    fn test_error_normalization() {
        // Test Error(Never) → Never
        let result = Type::make_error(Type::Never);
        assert_eq!(result, Type::Never);

        // Test Error(Error(T)) → Error(T)
        let inner_error = Type::make_error(Type::I32);
        let nested_error = Type::make_error(inner_error);
        let result = nested_error;
        assert_eq!(result, Type::make_error(Type::I32));
    }

    #[test]
    fn test_error_merging_in_unions() {
        // Test Union([Error(A), Error(B)]) → Union([Error(A), Error(B)]) (individual errors)
        let result = Type::make_union(vec![
            Type::make_error(Type::I32),
            Type::make_error(Type::Bool(BoolConstraint::Any)),
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                // Both should be individual Error types
                assert!(matches!(types[0], Type::Error(..)));
                assert!(matches!(types[1], Type::Error(..)));

                // Extract and verify inner types
                if let (Type::Error(inner1, ..), Type::Error(inner2, ..)) = (&types[0], &types[1]) {
                    let inner_types = vec![(&**inner1).clone(), (&**inner2).clone()];
                    assert!(inner_types.contains(&Type::Bool(BoolConstraint::Any)));
                    assert!(inner_types.contains(&Type::I32));
                }
            }
            _ => panic!("Expected Union with individual Error types"),
        }

        // Test mixed union: Union([A, Error(B)]) - errors sort to the end
        let result = Type::make_union(vec![
            Type::I32,
            Type::make_error(Type::Bool(BoolConstraint::Any)),
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                assert_eq!(types[0], Type::I32); // Regular type comes first
                assert!(matches!(types[1], Type::Error(..))); // Error type at tail

                if let Type::Error(inner, ..) = &types[1] {
                    assert_eq!(**inner, Type::Bool(BoolConstraint::Any));
                }
            }
            _ => panic!("Expected Union with mixed types"),
        }
    }

    #[test]
    fn test_error_intersect() {
        // Error(A) ∩ Error(B) → Error(A ∩ B)
        let error_i32 = Type::make_error(Type::I32);
        let error_bool = Type::make_error(Type::Bool(BoolConstraint::Any));
        let result = error_i32.intersect(&error_bool);
        assert_eq!(result, Type::Never); // I32 ∩ Bool = Never, so Error(Never) = Never

        // Error(A) ∩ Error(A) → Error(A)
        let error_i32_2 = Type::make_error(Type::I32);
        let result = error_i32.intersect(&error_i32_2);
        assert_eq!(result, Type::make_error(Type::I32));

        // Error(A) ∩ B → Never
        let result = error_i32.intersect(&Type::I32);
        assert_eq!(result, Type::Never);

        // A ∩ Error(B) → Never
        let result = Type::I32.intersect(&error_i32);
        assert_eq!(result, Type::Never);
    }

    #[test]
    fn test_bool_union_merging() {
        // Test Bool(Any) ∪ Bool(true) = Bool(Any)
        let result = Type::make_union(vec![
            Type::Bool(BoolConstraint::Any),
            Type::Bool(BoolConstraint::Const(true)),
        ]);
        assert_eq!(result, Type::Bool(BoolConstraint::Any));

        // Test Bool(true) ∪ Bool(false) = Bool(Any)
        let result = Type::make_union(vec![
            Type::Bool(BoolConstraint::Const(true)),
            Type::Bool(BoolConstraint::Const(false)),
        ]);
        assert_eq!(result, Type::Bool(BoolConstraint::Any));

        // Test Bool(true) ∪ Bool(true) = Bool(true)
        let result = Type::make_union(vec![
            Type::Bool(BoolConstraint::Const(true)),
            Type::Bool(BoolConstraint::Const(true)),
        ]);
        assert_eq!(result, Type::Bool(BoolConstraint::Const(true)));
    }

    #[test]
    fn test_float_union_merging() {
        use ordered_float::OrderedFloat;

        // Test Float(Any(F32)) ∪ Float(Const(2.5)) = Float(Any(F32))
        let result = Type::make_union(vec![
            Type::Float(FloatConstraint::Any(FloatPrim::F32)),
            Type::Float(FloatConstraint::Const(OrderedFloat(2.5))),
        ]);
        assert_eq!(result, Type::Float(FloatConstraint::Any(FloatPrim::F32)));

        // Test Float(Const(2.5)) ∪ Float(Const(2.5)) = Float(Const(2.5))
        let result = Type::make_union(vec![
            Type::Float(FloatConstraint::Const(OrderedFloat(2.5))),
            Type::Float(FloatConstraint::Const(OrderedFloat(2.5))),
        ]);
        assert_eq!(
            result,
            Type::Float(FloatConstraint::Const(OrderedFloat(2.5)))
        );

        // Test Float(Const(2.5)) ∪ Float(Const(3.7)) = Union (distinct constants stay separate)
        let result = Type::make_union(vec![
            Type::Float(FloatConstraint::Const(OrderedFloat(2.5))),
            Type::Float(FloatConstraint::Const(OrderedFloat(3.7))),
        ]);
        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                assert!(types.contains(&Type::Float(FloatConstraint::Const(OrderedFloat(2.5)))));
                assert!(types.contains(&Type::Float(FloatConstraint::Const(OrderedFloat(3.7)))));
            }
            _ => panic!("Expected union of distinct float constants"),
        }
    }

    #[test]
    fn test_error_subtract() {
        // With new flattening: make_error(Union([I32, Bool])) creates Union([Error(I32), Error(Bool)])
        // So we need to test subtraction on individual error types in the union
        let error_union = Type::make_error(Type::make_union(vec![
            Type::I32,
            Type::Bool(BoolConstraint::Any),
        ]));

        // This creates Union([Error(I32), Error(Bool)]) due to flattening
        match error_union {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                let error_i32 = Type::make_error(Type::I32);

                // Test subtract on the first error type (should be Error(I32) or Error(Bool))
                let result = types[0].subtract(&error_i32);
                // Error(I32) - Error(I32) = Never, Error(Bool) - Error(I32) = Error(Bool)
                if let Type::Error(inner, ..) = &types[0] {
                    if **inner == Type::I32 {
                        assert_eq!(result, Type::Never);
                    } else {
                        assert_eq!(result, types[0].clone()); // Error(Bool) unchanged
                    }
                }
            }
            _ => panic!("Expected Union type due to flattening"),
        }

        // Test simple error subtraction
        let error_i32 = Type::make_error(Type::I32);

        // Error(A) - B → Error(A) (unchanged)
        let result = error_i32.subtract(&Type::Bool(BoolConstraint::Any));
        assert_eq!(result, error_i32);

        // A - Error(B) → A (unchanged)
        let result = Type::I32.subtract(&error_i32);
        assert_eq!(result, Type::I32);

        // Error(A) - Error(A) → Never
        let result = error_i32.subtract(&error_i32);
        assert_eq!(result, Type::Never);
    }

    #[test]
    fn test_data_types() {
        use crate::symbols::{intern_symbol, symbol_name};

        // Test anonymous tuple
        let tuple_type = Type::make_tuple(vec![Type::I32, Type::Bool(BoolConstraint::Any)]);
        if let Type::Data(tag, fields) = tuple_type {
            assert_eq!(tag, None); // Anonymous
            assert_eq!(fields.len(), 2);
            assert_eq!(symbol_name(fields[0].name).unwrap(), "0");
            assert_eq!(symbol_name(fields[1].name).unwrap(), "1");
            assert_eq!(fields[0].ty, Type::I32);
            assert_eq!(fields[1].ty, Type::Bool(BoolConstraint::Any));
        } else {
            panic!("Expected Data type");
        }

        // Test named tuple
        let point_type = Type::make_named_tuple("Point", vec![Type::I32, Type::I32]);
        if let Type::Data(tag, fields) = point_type {
            assert_eq!(tag, Some(intern_symbol("Point")));
            assert_eq!(fields.len(), 2);
            assert_eq!(symbol_name(fields[0].name).unwrap(), "0");
            assert_eq!(symbol_name(fields[1].name).unwrap(), "1");
        } else {
            panic!("Expected Data type");
        }

        // Test anonymous record
        let record_type = Type::make_record(vec![("x", Type::I32), ("y", Type::I32)]);
        if let Type::Data(tag, fields) = record_type {
            assert_eq!(tag, None); // Anonymous
            assert_eq!(fields.len(), 2);
            assert_eq!(symbol_name(fields[0].name).unwrap(), "x");
            assert_eq!(symbol_name(fields[1].name).unwrap(), "y");
        } else {
            panic!("Expected Data type");
        }

        // Test named record
        let user_type = Type::make_named_record(
            "User",
            vec![
                ("name", Type::I32), // Using I32 for simplicity in tests
                ("age", Type::I32),
            ],
        );
        if let Type::Data(tag, fields) = user_type {
            assert_eq!(tag, Some(intern_symbol("User")));
            assert_eq!(fields.len(), 2);
            assert_eq!(symbol_name(fields[0].name).unwrap(), "name");
            assert_eq!(symbol_name(fields[1].name).unwrap(), "age");
        } else {
            panic!("Expected Data type");
        }

        // Test structural equality - same fields, same tag should be equal
        let point1 = Type::make_named_tuple("Point", vec![Type::I32, Type::I32]);
        let point2 = Type::make_named_tuple("Point", vec![Type::I32, Type::I32]);
        assert_eq!(point1, point2);

        // Test structural inequality - different tags
        let point = Type::make_named_tuple("Point", vec![Type::I32, Type::I32]);
        let vector = Type::make_named_tuple("Vector", vec![Type::I32, Type::I32]);
        assert_ne!(point, vector);
    }

    #[test]
    fn test_struct_coercion_intersect() {
        use crate::symbols::intern_symbol;

        // Create test fields
        let fields: Arc<[DataField]> = vec![
            DataField {
                name: intern_symbol("x"),
                ty: Type::I32,
            },
            DataField {
                name: intern_symbol("y"),
                ty: Type::I32,
            },
        ]
        .into();

        let untagged = Type::Data(None, fields.clone());
        let tagged_point = Type::Data(Some(intern_symbol("Point")), fields.clone());
        let tagged_vector = Type::Data(Some(intern_symbol("Vector")), fields.clone());

        // Untagged -> Tagged coercion
        let intersection = untagged.intersect(&tagged_point);
        assert_eq!(intersection, tagged_point);

        // Tagged -> Untagged coercion
        let intersection = tagged_point.intersect(&untagged);
        assert_eq!(intersection, untagged);

        // Different tags with same fields -> incompatible
        let intersection = tagged_point.intersect(&tagged_vector);
        assert_eq!(intersection, Type::Never);

        // Different fields -> incompatible
        let different_fields: Arc<[DataField]> = vec![DataField {
            name: intern_symbol("a"),
            ty: Type::I32,
        }]
        .into();
        let different_struct = Type::Data(None, different_fields);
        let intersection = untagged.intersect(&different_struct);
        assert_eq!(intersection, Type::Never);
    }

    #[test]
    fn test_struct_coercion_cast_kind() {
        use crate::symbols::intern_symbol;

        let fields: Arc<[DataField]> = vec![
            DataField {
                name: intern_symbol("x"),
                ty: Type::I32,
            },
            DataField {
                name: intern_symbol("y"),
                ty: Type::I32,
            },
        ]
        .into();

        let untagged = Type::Data(None, fields.clone());
        let tagged_point = Type::Data(Some(intern_symbol("Point")), fields.clone());

        // Untagged can cast to tagged (static)
        assert_eq!(untagged.cast_kind(&tagged_point), CastKind::Static);

        // Tagged can cast to untagged (static)
        assert_eq!(tagged_point.cast_kind(&untagged), CastKind::Static);

        // Different tags cannot cast
        let tagged_vector = Type::Data(Some(intern_symbol("Vector")), fields.clone());
        assert_eq!(tagged_point.cast_kind(&tagged_vector), CastKind::Invalid);
    }

    #[test]
    fn test_union_coercion_ambiguity() {
        use crate::symbols::intern_symbol;

        let fields: Arc<[DataField]> = vec![
            DataField {
                name: intern_symbol("x"),
                ty: Type::I32,
            },
            DataField {
                name: intern_symbol("y"),
                ty: Type::I32,
            },
        ]
        .into();

        let untagged = Type::Data(None, fields.clone());
        let tagged_point = Type::Data(Some(intern_symbol("Point")), fields.clone());
        let tagged_vector = Type::Data(Some(intern_symbol("Vector")), fields.clone());

        // Single match - should succeed
        let union_single = Type::make_union(vec![tagged_point.clone(), Type::I32]);
        let intersection = untagged.intersect(&union_single);
        assert_eq!(intersection, tagged_point);

        // Multiple tagged matches - should fail (ambiguous)
        let union_ambiguous = Type::make_union(vec![tagged_point.clone(), tagged_vector.clone()]);
        let intersection = untagged.intersect(&union_ambiguous);
        assert_eq!(intersection, Type::Never);

        // Tagged + untagged - should return exact match (untagged)
        let union_mixed = Type::make_union(vec![tagged_point.clone(), untagged.clone()]);
        let intersection = untagged.intersect(&union_mixed);
        assert_eq!(intersection, untagged); // Exact match wins over specificity
    }

    #[test]
    fn test_union_intersection_commutativity() {
        use crate::symbols::intern_symbol;

        let fields: Arc<[DataField]> = vec![
            DataField {
                name: intern_symbol("x"),
                ty: Type::I32,
            },
            DataField {
                name: intern_symbol("y"),
                ty: Type::I32,
            },
        ]
        .into();

        let untagged = Type::Data(None, fields.clone());
        let tagged_point = Type::Data(Some(intern_symbol("Point")), fields.clone());
        let union = Type::make_union(vec![tagged_point.clone(), Type::I32]);

        // Test commutativity: a ∩ b = b ∩ a
        let intersection1 = untagged.intersect(&union);
        let intersection2 = union.intersect(&untagged);
        assert_eq!(intersection1, intersection2);
        assert_eq!(intersection1, tagged_point);
    }
}
