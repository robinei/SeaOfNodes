use std::sync::Arc;

use ordered_float::OrderedFloat;

use crate::IRError;
use crate::constraints::*;

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

    Union(Arc<[Type]>, Priv),

    Error(Arc<Type>, Priv),
}

impl Type {
    /// Try to merge two types in a union. Returns Some(merged) if possible, None if not.
    fn try_merge_types(left: &Type, right: &Type) -> Option<Type> {
        match (left, right) {
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
            (Type::Float(FloatConstraint::Any(prim)), Type::Float(FloatConstraint::Const(_)))
            | (Type::Float(FloatConstraint::Const(_)), Type::Float(FloatConstraint::Any(prim))) => {
                // Polymorphic constant gets subsumed by specific primitive
                Some(Type::Float(FloatConstraint::Any(*prim)))
            }
            (Type::Float(FloatConstraint::Const(a)), Type::Float(FloatConstraint::Const(b)))
                if a == b =>
            {
                // Identical constants merge
                Some(Type::Float(FloatConstraint::Const(*a)))
            }
            (Type::Error(linner, ..), Type::Error(rinner, ..)) => {
                // Always merge adjacent errors by creating Error(Union([contents]))
                let merged_contents = Type::make_union(vec![
                    Arc::try_unwrap(linner.clone()).unwrap_or_else(|arc| (*arc).clone()),
                    Arc::try_unwrap(rinner.clone()).unwrap_or_else(|arc| (*arc).clone()),
                ]);
                Some(Type::Error(Arc::new(merged_contents), Priv::new()))
            }
            _ => None, // Can't merge
        }
    }

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

    pub fn make_union(mut types: Vec<Type>) -> Type {
        // Flatten nested unions in-place
        let mut i = 0;
        while i < types.len() {
            if let Type::Union(..) = &types[i] {
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

    pub fn make_error(inner: Type) -> Type {
        match inner {
            Type::Never => Type::Never,
            Type::Error(..) => inner,
            Type::Union(types, ..) => {
                // Distribute: Error(Union([A, B])) → make_union([Error(A), Error(B)])
                let error_types: Vec<Type> =
                    types.iter().map(|t| Type::make_error(t.clone())).collect();
                Type::make_union(error_types) // Will handle Error merging
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
        match (self, other) {
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

            (Type::Bool(lc), Type::Bool(rc)) => match (lc, rc) {
                (BoolConstraint::Const(a), BoolConstraint::Const(b)) if a == b => self.clone(),
                (BoolConstraint::Const(_), BoolConstraint::Const(_)) => Type::Never,
                (BoolConstraint::Const(_), BoolConstraint::Any) => self.clone(),
                (BoolConstraint::Any, BoolConstraint::Const(_)) => other.clone(),
                (BoolConstraint::Any, BoolConstraint::Any) => self.clone(),
            },

            // Error intersections
            (Type::Error(la, ..), Type::Error(ra, ..)) => {
                let inner_intersection = (**la).clone().intersect(&(**ra));
                match inner_intersection {
                    Type::Never => Type::Never,
                    other => Type::make_error(other),
                }
            }

            (Type::Error(..), _) => Type::Never, // Error ∩ non-error = Never
            (_, Type::Error(..)) => Type::Never, // non-error ∩ Error = Never

            // Union intersections
            (Type::Union(types, ..), other) => {
                let intersected: Vec<Type> = types
                    .iter()
                    .map(|t| t.intersect(other))
                    .filter(|t| !matches!(t, Type::Never))
                    .collect();
                Type::make_union(intersected)
            }

            (other, Type::Union(types, ..)) => other.intersect(&Type::make_union(types.to_vec())),

            // Exact same type → return self
            (a, b) if a == b => self.clone(),

            // Different types → no intersection
            _ => Type::Never,
        }
    }

    pub fn subtract(&self, other: &Self) -> Type {
        match (self, other) {
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

            (Type::Bool(lc), Type::Bool(rc)) => {
                match (lc, rc) {
                    (BoolConstraint::Const(a), BoolConstraint::Const(b)) if a == b => Type::Never,
                    (BoolConstraint::Const(_), BoolConstraint::Const(_)) => self.clone(), // Different constants
                    (BoolConstraint::Any, BoolConstraint::Const(b)) => Type::const_bool(!b),
                    (BoolConstraint::Const(_), BoolConstraint::Any) => Type::Never,
                    (BoolConstraint::Any, BoolConstraint::Any) => Type::Never,
                }
            }

            // Error subtraction
            (Type::Error(la, ..), Type::Error(ra, ..)) => {
                let inner_subtraction = (**la).clone().subtract(&(**ra));
                match inner_subtraction {
                    Type::Never => Type::Never,
                    other => Type::make_error(other),
                }
            }

            (Type::Error(..), _) => self.clone(), // Error - non-error = Error (unchanged)
            (_, Type::Error(..)) => self.clone(), // non-error - Error = non-error (unchanged)

            // Union subtraction
            (Type::Union(types, ..), other) => {
                let subtracted: Vec<Type> = types
                    .iter()
                    .map(|t| t.subtract(other))
                    .filter(|t| !matches!(t, Type::Never))
                    .collect();
                Type::make_union(subtracted)
            }

            // Exact same type → empty result
            (a, b) if a == b => Type::Never,

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
            (Type::Int(lp, lc), Type::Int(rp, rc)) if lp == rp => {
                Ok(Type::bool_from_option(lc.compare(*rc, op)))
            }
            (Type::UInt(lp, lc), Type::UInt(rp, rc)) if lp == rp => {
                Ok(Type::bool_from_option(lc.compare(*rc, op)))
            }
            (Type::Float(lc), Type::Float(rc)) => Ok(Type::bool_from_option(lc.compare(*rc, op))),
            (Type::Bool(lc), Type::Bool(rc)) => Ok(Type::bool_from_option(lc.compare(*rc, op))),
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
        // Test Union([Error(A), Error(B)]) → Error(Union([A, B]))
        let result = Type::make_union(vec![
            Type::make_error(Type::I32),
            Type::make_error(Type::Bool(BoolConstraint::Any)),
        ]);

        match result {
            Type::Error(inner, ..) => match &*inner {
                Type::Union(types, ..) => {
                    assert_eq!(types.len(), 2);
                    assert!(types.contains(&Type::Bool(BoolConstraint::Any)));
                    assert!(types.contains(&Type::I32));
                }
                _ => panic!("Expected Error(Union(...))"),
            },
            _ => panic!("Expected Error type"),
        }

        // Test mixed union: Union([A, Error(B)]) stays as-is
        let result = Type::make_union(vec![
            Type::I32,
            Type::make_error(Type::Bool(BoolConstraint::Any)),
        ]);

        match result {
            Type::Union(types, ..) => {
                assert_eq!(types.len(), 2);
                assert!(types.contains(&Type::I32));
                assert!(types.contains(&Type::make_error(Type::Bool(BoolConstraint::Any))));
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
        // Error(A) - Error(B) → Error(A - B)
        let error_union = Type::make_error(Type::make_union(vec![
            Type::I32,
            Type::Bool(BoolConstraint::Any),
        ]));
        let error_i32 = Type::make_error(Type::I32);
        let result = error_union.subtract(&error_i32);

        match result {
            Type::Error(inner, ..) => {
                assert_eq!(*inner, Type::Bool(BoolConstraint::Any)); // Union([I32, Bool]) - I32 = Bool
            }
            _ => panic!("Expected Error type"),
        }

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
}
