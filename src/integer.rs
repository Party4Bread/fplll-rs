//! Multi-precision integer type. Wraps [`rug::Integer`].
//!
//! Mirrors fplll's `Z_NR<mpz_t>`. Operations are provided as methods so the
//! call-site style matches the C++ library (addmul, submul, mul_2si, etc.).

use rug::{Assign, Integer as RugInt};
use std::fmt;
use std::ops::AddAssign;

pub type Integer = RugInt;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Z(pub Integer);

impl Z {
    pub fn new() -> Self { Z(Integer::new()) }
    pub fn from_i64(v: i64) -> Self { Z(Integer::from(v)) }
    pub fn from_u64(v: u64) -> Self { Z(Integer::from(v)) }
    pub fn zero() -> Self { Z(Integer::new()) }
    pub fn one() -> Self { Z(Integer::from(1)) }

    pub fn is_zero(&self) -> bool { self.0 == 0 }
    pub fn sign(&self) -> i32 {
        match self.0.cmp0() {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
    pub fn abs_inplace(&mut self) { self.0.abs_mut(); }
    pub fn neg_inplace(&mut self) { self.0 = -std::mem::take(&mut self.0); }
    pub fn get_si(&self) -> i64 { self.0.to_i64_wrapping() }
    pub fn get_d(&self) -> f64 { self.0.to_f64() }

    /// f * 2^expo = self, with f in [0.5, 1) (like mpz_get_d_2exp)
    pub fn get_d_2exp(&self) -> (f64, i64) {
        // rug provides Integer::to_f64_exp-like behaviour via get_d_2exp on mpz.
        // rug exposes this via `to_f64` and `significant_bits`. Emulate:
        if self.0 == 0 { return (0.0, 0); }
        let bits = self.0.significant_bits() as i64;
        let exp = bits;
        let mut scaled = self.0.clone();
        // shift so magnitude is in [2^52, 2^53), then convert to f64.
        let shift = bits as i32 - 53;
        if shift > 0 {
            scaled >>= shift as u32;
        } else if shift < 0 {
            scaled <<= (-shift) as u32;
        }
        let v = scaled.to_f64() / (1u64 << 53) as f64;
        (v, exp)
    }

    pub fn add_assign_ref(&mut self, other: &Z) { self.0 += &other.0; }
    pub fn sub_assign_ref(&mut self, other: &Z) { self.0 -= &other.0; }
    pub fn mul_assign_ref(&mut self, other: &Z) { self.0 *= &other.0; }

    /// self = a * b
    pub fn mul_set(&mut self, a: &Z, b: &Z) {
        self.0.assign(&a.0 * &b.0);
    }
    /// self = a + b
    pub fn add_set(&mut self, a: &Z, b: &Z) {
        self.0.assign(&a.0 + &b.0);
    }
    /// self = a - b
    pub fn sub_set(&mut self, a: &Z, b: &Z) {
        self.0.assign(&a.0 - &b.0);
    }
    /// self += a * b
    pub fn addmul(&mut self, a: &Z, b: &Z) {
        self.0.add_assign(&a.0 * &b.0);
    }
    /// self -= a * b
    pub fn submul(&mut self, a: &Z, b: &Z) {
        self.0 -= &a.0 * &b.0;
    }
    /// self += a * x
    pub fn addmul_si(&mut self, a: &Z, x: i64) {
        self.0.add_assign(&a.0 * x);
    }
    /// self -= a * x
    pub fn submul_si(&mut self, a: &Z, x: i64) {
        self.0 -= &a.0 * x;
    }
    /// self = a * x
    pub fn mul_si(&mut self, a: &Z, x: i64) {
        self.0.assign(&a.0 * x);
    }
    /// self = a * 2^e
    pub fn mul_2si(&mut self, a: &Z, e: i64) {
        if e >= 0 {
            self.0.assign(&a.0 << e as u32);
        } else {
            // Arithmetic right shift for signed MPZ: rug does floor-div by 2^n.
            self.0.assign(&a.0 >> ((-e) as u32));
        }
    }
    /// self += a * 2^e
    pub fn addmul_2si(&mut self, a: &Z, e: i64, tmp: &mut Z) {
        tmp.mul_2si(a, e);
        self.0 += &tmp.0;
    }

    pub fn cmp(&self, other: &Z) -> std::cmp::Ordering { self.0.cmp(&other.0) }
    pub fn cmp_si(&self, x: i64) -> std::cmp::Ordering { self.0.cmp(&RugInt::from(x)) }
    pub fn set(&mut self, other: &Z) { self.0.assign(&other.0); }
    pub fn set_si(&mut self, x: i64) { self.0.assign(x); }

    pub fn next_prime(&self) -> Z {
        let mut r = self.0.clone();
        r.next_prime_mut();
        Z(r)
    }

    pub fn to_str_radix(&self, radix: i32) -> String {
        self.0.to_string_radix(radix)
    }

    pub fn from_str(s: &str) -> Option<Z> {
        RugInt::parse(s).ok().map(|p| Z(p.complete()))
    }
}

impl Default for Z { fn default() -> Self { Z::new() } }

impl fmt::Display for Z {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(&self.0, f) }
}

impl fmt::Debug for Z {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(&self.0, f) }
}

impl From<i64> for Z { fn from(v: i64) -> Self { Z::from_i64(v) } }
impl From<i32> for Z { fn from(v: i32) -> Self { Z::from_i64(v as i64) } }
impl From<u64> for Z { fn from(v: u64) -> Self { Z::from_u64(v) } }
impl From<Integer> for Z { fn from(v: Integer) -> Self { Z(v) } }

impl std::cmp::PartialOrd for Z {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl std::cmp::Ord for Z {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0) }
}

use rug::Complete;
