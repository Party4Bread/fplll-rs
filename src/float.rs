//! Floating point scalar abstraction, modelled after fplll's FP_NR.
//!
//! This port uses `f64` for the primary floating point type (matching
//! fplll's `FT_DOUBLE`, the workhorse of most algorithms). The type alias
//! exists so future backends (long double, mpfr via rug::Float) can be
//! swapped in without touching call sites.

pub type FloatScalar = f64;

/// Round to nearest integer, ties to even.
#[inline]
pub fn rnd(x: f64) -> f64 { x.round_ties_even() }

/// Round to nearest i64.
#[inline]
pub fn rnd_i64(x: f64) -> i64 { rnd(x) as i64 }

/// Largest `e` such that `|x| * 2^-e` fits a double without loss (proxy for FP_NR::exponent).
#[inline]
pub fn exponent(x: f64) -> i64 {
    if x == 0.0 { return 0; }
    // IEEE-754 binary64 exponent bias is 1023. Rust's frexp equivalent:
    let (_m, e) = frexp(x);
    e as i64
}

/// Equivalent of libc frexp.
#[inline]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() { return (x, 0); }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let (m_bits, e) = if exp_bits == 0 {
        // Subnormal
        let shift = mantissa.leading_zeros() as i32 - 11;
        let m = mantissa << shift;
        (m, -1022 - shift)
    } else {
        (mantissa, exp_bits - 1022)
    };
    let new_bits = sign | ((1022u64) << 52) | (m_bits & 0x000f_ffff_ffff_ffff);
    (f64::from_bits(new_bits), e)
}

/// Equivalent of libc ldexp.
#[inline]
pub fn ldexp(x: f64, e: i32) -> f64 { x * 2f64.powi(e) }

/// Round-then-extract: set `*dest = round(src * 2^-expo)` (mpfr-like rnd_we).
#[inline]
pub fn rnd_we(src: f64, expo: i32) -> f64 {
    rnd(ldexp(src, -expo))
}
