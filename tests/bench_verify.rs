//! Bench + correctness verify: measure LLL and check the output is LLL-reduced.

use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::lll::{is_lll_reduced_basis, lll_reduce};
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;
use std::time::Instant;

#[test]
fn lll_1024bit_20dim_verified() {
    for seed in 0..5u64 {
        let mut rng = RandCtx::new_seeded(42 + seed);
        let mut m = ZMatrix::new();
        m.resize(20, 21);
        m.gen_intrel(1024, &mut rng);
        let t = Instant::now();
        let st = lll_reduce(&mut m, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
        let dt = t.elapsed().as_secs_f64();
        let ok = is_lll_reduced_basis(&m, LLL_DEF_DELTA, LLL_DEF_ETA);
        eprintln!("seed {:>3}: {:.4}s  status={:?}  lll_reduced={}", 42 + seed, dt, st, ok);
        assert_eq!(st as i32, 0);
        assert!(ok, "output of LLL is not actually LLL-reduced for seed {}", 42 + seed);
    }
}
