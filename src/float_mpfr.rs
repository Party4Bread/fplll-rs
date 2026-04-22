//! MPFR-backed floating point via `rug::Float`.
//!
//! This provides an arbitrary-precision float alternative to `f64` so the
//! GSO/LLL code can reduce bases whose entries or GS norms exceed the
//! dynamic range of double precision. Precision is set per-instance.

use rug::float::Round;
use rug::{Assign, Float};

pub const DEFAULT_PREC: u32 = 128;

thread_local! {
    static PREC: std::cell::Cell<u32> = std::cell::Cell::new(DEFAULT_PREC);
}

pub fn set_prec(p: u32) -> u32 {
    PREC.with(|c| { let old = c.get(); c.set(p); old })
}
pub fn get_prec() -> u32 { PREC.with(|c| c.get()) }

/// Construct a fresh zero at the current precision.
#[inline] pub fn zero() -> Float { Float::with_val(get_prec(), 0) }

#[inline] pub fn from_f64(v: f64) -> Float { Float::with_val(get_prec(), v) }

#[inline] pub fn from_i64(v: i64) -> Float { Float::with_val(get_prec(), v) }

/// `dest = a * b`
#[inline] pub fn mul(dest: &mut Float, a: &Float, b: &Float) { dest.assign(a * b); }
/// `dest = a + b`
#[inline] pub fn add(dest: &mut Float, a: &Float, b: &Float) { dest.assign(a + b); }
/// `dest = a - b`
#[inline] pub fn sub(dest: &mut Float, a: &Float, b: &Float) { dest.assign(a - b); }
/// `dest = a / b`
#[inline] pub fn div(dest: &mut Float, a: &Float, b: &Float) { dest.assign(a / b); }

/// `dest += a * b`
#[inline] pub fn addmul(dest: &mut Float, a: &Float, b: &Float) {
    let prod = Float::with_val(get_prec(), a * b);
    *dest += prod;
}

/// `dest -= a * b`
#[inline] pub fn submul(dest: &mut Float, a: &Float, b: &Float) {
    let prod = Float::with_val(get_prec(), a * b);
    *dest -= prod;
}

/// Round to nearest integer as i64 (saturating if huge).
pub fn to_i64_rnd(x: &Float) -> i64 {
    if x.is_zero() { return 0; }
    to_integer_rnd(x).to_i64_wrapping()
}

/// Round to nearest integer as rug::Integer.
pub fn to_integer_rnd(x: &Float) -> rug::Integer {
    x.to_integer_round(Round::Nearest).map(|t| t.0).unwrap_or_default()
}

/// Convert to f64 (may lose precision / saturate to inf).
#[inline] pub fn to_f64(x: &Float) -> f64 { x.to_f64() }

/// Compare, returning Ordering.
#[inline] pub fn cmp(a: &Float, b: &Float) -> std::cmp::Ordering { a.partial_cmp(b).unwrap() }

/// Absolute value (new).
#[inline] pub fn abs(x: &Float) -> Float { x.clone().abs() }
