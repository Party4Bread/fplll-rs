//! BKZ tests (port of fplll/tests/test_bkz.cpp subset).

use fplll::bkz::{bkz_reduce, BkzParam};
use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::lll::is_lll_reduced_basis;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn bkz_reduces_uniform() {
    let mut rng = RandCtx::new_seeded(11);
    let mut b = ZMatrix::new();
    b.resize(10, 10);
    b.gen_uniform(20, &mut rng);
    let p = BkzParam { block_size: 4, ..Default::default() };
    let st = bkz_reduce(&mut b, &p);
    assert_eq!(st as i32, 0);
    // BKZ-reduced implies LLL-reduced.
    assert!(is_lll_reduced_basis(&b, LLL_DEF_DELTA, LLL_DEF_ETA));
}

#[test]
fn bkz_intrel() {
    let mut rng = RandCtx::new_seeded(5);
    let mut b = ZMatrix::new();
    b.resize(12, 13);
    b.gen_intrel(100, &mut rng);
    let p = BkzParam { block_size: 6, ..Default::default() };
    let st = bkz_reduce(&mut b, &p);
    assert_eq!(st as i32, 0);
    assert!(is_lll_reduced_basis(&b, LLL_DEF_DELTA, LLL_DEF_ETA));
}
