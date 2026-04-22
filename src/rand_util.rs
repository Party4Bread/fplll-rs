//! Random number generation (mirrors fplll's RandGen).
//!
//! fplll uses a single global gmp randstate; we keep an equivalent context
//! so tests can be seeded deterministically.

use rug::rand::RandState;
use rug::Integer;

pub struct RandCtx {
    state: RandState<'static>,
}

impl RandCtx {
    pub fn new_seeded(seed: u64) -> Self {
        let mut state = RandState::new();
        state.seed(&Integer::from(seed));
        RandCtx { state }
    }
    pub fn new_with_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self::new_seeded(nanos ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn state(&mut self) -> &mut RandState<'static> { &mut self.state }
}

/// Uniformly random integer in [0, 2^bits).
pub fn rand_bits(ctx: &mut RandCtx, bits: u32) -> Integer {
    Integer::random_bits(bits, ctx.state()).into()
}

/// Uniformly random integer in [0, bound).
pub fn rand_below(ctx: &mut RandCtx, bound: &Integer) -> Integer {
    bound.clone().random_below(ctx.state())
}

/// Random i64 in [lo, hi].
pub fn rand_range_i64(ctx: &mut RandCtx, lo: i64, hi: i64) -> i64 {
    if hi <= lo { return lo; }
    let span = (hi - lo + 1) as u64;
    let r = Integer::from(span).random_below(ctx.state()).to_u64_wrapping();
    lo + r as i64
}

thread_local! {
    static DEFAULT_CTX: std::cell::RefCell<RandCtx> = std::cell::RefCell::new(RandCtx::new_seeded(0xfc0ff));
}

pub fn with_default<R>(f: impl FnOnce(&mut RandCtx) -> R) -> R {
    DEFAULT_CTX.with(|c| f(&mut *c.borrow_mut()))
}
