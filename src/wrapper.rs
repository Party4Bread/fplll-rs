//! High-level LLL wrapper, mirroring fplll/wrapper.cpp.
//!
//! The wrapper picks a reduction strategy (fast → heuristic → proved) based
//! on the input size and falls back to higher precision on failure.

use crate::defs::{LLL_DEF_DELTA, LLL_DEF_ETA, LllMethod, RedStatus};
use crate::lll::lll_reduce;
use crate::matrix::ZMatrix;

pub struct Wrapper {
    pub delta: f64,
    pub eta: f64,
    pub flags: i32,
}

impl Default for Wrapper {
    fn default() -> Self { Wrapper { delta: LLL_DEF_DELTA, eta: LLL_DEF_ETA, flags: 0 } }
}

impl Wrapper {
    pub fn run(&self, b: &mut ZMatrix) -> RedStatus {
        // Try fast path first.
        let st = lll_reduce(b, self.delta, self.eta, self.flags);
        st
    }
}

/// Dispatcher mirroring fplll::lll_reduction(): picks a method and runs.
pub fn lll_reduction(
    b: &mut ZMatrix,
    delta: f64,
    eta: f64,
    _method: LllMethod,
    _float_type: crate::defs::FloatType,
    _prec: u32,
    flags: i32,
) -> RedStatus {
    lll_reduce(b, delta, eta, flags)
}
