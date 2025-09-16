use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, One, Zero};
use ordered_float::OrderedFloat;
use crate::IRError;

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntPrim {
    I8,
    I16,
    I32,
    I64,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UIntPrim {
    U8,
    U16,
    U32,
    U64,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatPrim {
    F32,
    F64,
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

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CompareOp {
    Lt,   // <
    Gt,   // >
    LtEq, // <=
    GtEq, // >=
    Eq,   // ==
    Neq,  // !=
}

pub trait ConstraintValue:
    Copy + Clone + PartialOrd + Ord + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Zero + One + std::fmt::Debug
{
    type Primitive: Copy + Clone + PartialEq + Eq + std::fmt::Debug;

    // Get min/max values for a specific primitive
    fn prim_min(prim: Self::Primitive) -> Self;
    fn prim_max(prim: Self::Primitive) -> Self;

    // Check if this primitive can represent untyped constants
    fn is_const_storage(prim: Self::Primitive) -> bool;
}

impl ConstraintValue for i64 {
    type Primitive = IntPrim;

    fn prim_min(prim: IntPrim) -> Self {
        prim.min()
    }

    fn prim_max(prim: IntPrim) -> Self {
        prim.max()
    }

    fn is_const_storage(prim: IntPrim) -> bool {
        prim == IntPrim::I64 // Only i64 can represent untyped signed constants
    }
}

impl ConstraintValue for u64 {
    type Primitive = UIntPrim;

    fn prim_min(prim: UIntPrim) -> Self {
        prim.min()
    }

    fn prim_max(prim: UIntPrim) -> Self {
        prim.max()
    }

    fn is_const_storage(prim: UIntPrim) -> bool {
        prim == UIntPrim::U64 // Only u64 can represent untyped unsigned constants
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RangeConstraint<T: ConstraintValue> {
    pub min: T,
    pub max: T,
}

impl<T: ConstraintValue> RangeConstraint<T> {
    #[inline]
    pub const fn new(min: T, max: T) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn is_const(&self) -> bool {
        self.min == self.max
    }

    #[inline]
    pub fn get_const_value(&self) -> Option<T> {
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
        lp: T::Primitive,
        rp: T::Primitive,
        op: impl Fn(Self, Self) -> Option<(T, T)>,
    ) -> Result<(T::Primitive, Self), IRError> {
        // Resolve result primitive type
        let result_prim = match (lp, rp) {
            (l, r) if l == r => l,
            // Allow any constant to coerce to a specific primitive type
            (l, _r) if other.is_const() => l,
            (_l, r) if self.is_const() => r,
            _ => return Err(IRError::InvalidPrimitiveCoercion),
        };


        // Apply operation to get min/max bounds
        let constraint = match op(self, other) {
            Some((min, max)) => {
                let clamped_min = min.max(T::prim_min(result_prim));
                let clamped_max = max.min(T::prim_max(result_prim));

                // Error only if the entire range is outside the target primitive bounds
                if max < T::prim_min(result_prim) || min > T::prim_max(result_prim) {
                    return Err(IRError::IntegerOverflow); // Definite overflow - entire range is out of bounds
                }

                Self::new(clamped_min, clamped_max)
            }
            None => {
                // Fall back to full range - don't error even with constants
                // because None just means "can't determine precise range"
                Self::new(T::prim_min(result_prim), T::prim_max(result_prim))
            }
        };

        Ok((result_prim, constraint))
    }

    #[inline]
    fn apply_4way_op(self, other: Self, op: impl Fn(T, T) -> Option<T>) -> Option<(T, T)> {
        let results = [
            op(self.min, other.min),
            op(self.min, other.max),
            op(self.max, other.min),
            op(self.max, other.max),
        ];

        if let [Some(r0), Some(r1), Some(r2), Some(r3)] = results {
            let mut min = r0;
            let mut max = r0;
            
            if r1 < min { min = r1; }
            if r1 > max { max = r1; }
            if r2 < min { min = r2; }
            if r2 > max { max = r2; }
            if r3 < min { min = r3; }
            if r3 > max { max = r3; }
            
            Some((min, max))
        } else {
            None
        }
    }

    pub fn add(
        self,
        other: Self,
        lp: T::Primitive,
        rp: T::Primitive,
    ) -> Result<(T::Primitive, Self), IRError> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((a.min.checked_add(&b.min)?, a.max.checked_add(&b.max)?))
        })
    }

    pub fn sub(
        self,
        other: Self,
        lp: T::Primitive,
        rp: T::Primitive,
    ) -> Result<(T::Primitive, Self), IRError> {
        self.combine_constraints(other, lp, rp, |a, b| {
            Some((a.min.checked_sub(&b.max)?, a.max.checked_sub(&b.min)?))
        })
    }

    pub fn mul(
        self,
        other: Self,
        lp: T::Primitive,
        rp: T::Primitive,
    ) -> Result<(T::Primitive, Self), IRError> {
        self.combine_constraints(other, lp, rp, |a, b| {
            a.apply_4way_op(b, |x, y| x.checked_mul(&y))
        })
    }

    pub fn div(
        self,
        other: Self,
        lp: T::Primitive,
        rp: T::Primitive,
    ) -> Result<(T::Primitive, Self), IRError> {
        // Only error if we're definitely dividing by zero (constant zero)
        if other.is_const() && other.get_const_value() == Some(T::zero()) {
            return Err(IRError::DivisionByZero);
        }

        self.combine_constraints(other, lp, rp, |a, b| {
            // For division, we need to handle zero in the denominator carefully
            if b.min <= T::zero() && b.max >= T::zero() {
                // Denominator range includes zero - we can't use apply_4way_op directly
                // Instead, we need to split the range or fall back to full range
                return None; // Fallback to full range for now
            }
            a.apply_4way_op(b, |x, y| x.checked_div(&y))
        })
    }

    pub fn compare(self, other: Self, op: CompareOp) -> Option<bool> {
        match op {
            CompareOp::Lt => {
                if self.max < other.min {
                    Some(true)
                } else if self.min >= other.max {
                    Some(false)
                } else {
                    None
                }
            }
            CompareOp::Gt => {
                if self.min > other.max {
                    Some(true)
                } else if self.max <= other.min {
                    Some(false)
                } else {
                    None
                }
            }
            CompareOp::LtEq => {
                if self.max <= other.min {
                    Some(true)
                } else if self.min > other.max {
                    Some(false)
                } else {
                    None
                }
            }
            CompareOp::GtEq => {
                if self.min >= other.max {
                    Some(true)
                } else if self.max < other.min {
                    Some(false)
                } else {
                    None
                }
            }
            CompareOp::Eq => {
                if self.max < other.min || other.max < self.min {
                    Some(false) // Disjoint
                } else if self.is_const() && other.is_const() {
                    Some(self.get_const_value() == other.get_const_value())
                } else if self == other {
                    Some(true) // Identical ranges
                } else {
                    None
                }
            }
            CompareOp::Neq => {
                if self.max < other.min || other.max < self.min {
                    Some(true) // Disjoint → always not equal
                } else if self.is_const() && other.is_const() {
                    Some(self.get_const_value() != other.get_const_value())
                } else if self == other {
                    Some(false) // Identical ranges → never not equal
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    pub fn compare(self, other: Self, op: CompareOp) -> Option<bool> {
        match (self, other) {
            (BoolConstraint::Const(a), BoolConstraint::Const(b)) => Some(match op {
                CompareOp::Lt => a < b,
                CompareOp::Gt => a > b,
                CompareOp::LtEq => a <= b,
                CompareOp::GtEq => a >= b,
                CompareOp::Eq => a == b,
                CompareOp::Neq => a != b,
            }),
            _ => None, // Conservative for non-constant bools
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatConstraint {
    Any(FloatPrim),
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
                FloatConstraint::Any(FloatPrim::F32) | FloatConstraint::Const(..),
                FloatConstraint::Any(FloatPrim::F32) | FloatConstraint::Const(..),
            ) => FloatConstraint::Any(FloatPrim::F32),
            (
                FloatConstraint::Any(FloatPrim::F64) | FloatConstraint::Const(..),
                FloatConstraint::Any(FloatPrim::F64) | FloatConstraint::Const(..),
            ) => FloatConstraint::Any(FloatPrim::F64),
            _ => unreachable!("bad type combination"),
        }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval + rval)
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval - rval)
    }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval * rval)
    }

    #[inline]
    pub fn div(self, other: Self) -> Self {
        self.combine(other, |lval, rval| lval / rval)
    }

    pub fn compare(self, other: Self, op: CompareOp) -> Option<bool> {
        match (self, other) {
            (FloatConstraint::Const(a), FloatConstraint::Const(b)) => Some(match op {
                CompareOp::Lt => a < b,
                CompareOp::Gt => a > b,
                CompareOp::LtEq => a <= b,
                CompareOp::GtEq => a >= b,
                CompareOp::Eq => a == b,
                CompareOp::Neq => a != b,
            }),
            _ => None, // Conservative for non-constant floats
        }
    }
}

// Type aliases for convenience
pub type IntConstraint = RangeConstraint<i64>;
pub type UIntConstraint = RangeConstraint<u64>;

// Common range type for comparison across signed/unsigned boundaries
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CommonRange {
    pub min: i128,
    pub max: i128,
}

impl CommonRange {
    pub fn new(min: i128, max: i128) -> Self {
        Self { min, max }
    }
    
    /// Check if self completely contains other
    pub fn contains(&self, other: &Self) -> bool {
        self.min <= other.min && self.max >= other.max
    }
    
    /// Check if self overlaps with other
    pub fn overlaps(&self, other: &Self) -> bool {
        self.max >= other.min && self.min <= other.max
    }
    
    /// Check if self is disjoint from other
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.overlaps(other)
    }
}

impl From<IntConstraint> for CommonRange {
    fn from(constraint: IntConstraint) -> Self {
        Self::new(constraint.min as i128, constraint.max as i128)
    }
}

impl From<UIntConstraint> for CommonRange {
    fn from(constraint: UIntConstraint) -> Self {
        Self::new(constraint.min as i128, constraint.max as i128)
    }
}
