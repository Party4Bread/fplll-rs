//! HLLL tests (port of fplll/tests/test_hlll.cpp subset).

use fplll::defs::{HLLL_DEF_C, HLLL_DEF_THETA, LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::hlll::hlll_reduce;
use fplll::lll::is_lll_reduced_basis;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn hlll_reduces_uniform_small() {
    let mut rng = RandCtx::new_seeded(13);
    let mut b = ZMatrix::new();
    b.resize(6, 6);
    b.gen_uniform(20, &mut rng);
    let st = hlll_reduce(&mut b, LLL_DEF_DELTA, LLL_DEF_ETA);
    assert_eq!(st as i32, 0);
    // HLLL-output is also (delta,eta)-LLL reduced.
    assert!(is_lll_reduced_basis(&b, LLL_DEF_DELTA, LLL_DEF_ETA));
    let _ = (HLLL_DEF_C, HLLL_DEF_THETA);
}

#[test]
fn hlll_reduces_intrel_small() {
    let mut rng = RandCtx::new_seeded(41);
    let mut b = ZMatrix::new();
    b.resize(8, 9);
    b.gen_intrel(20, &mut rng);
    let st = hlll_reduce(&mut b, LLL_DEF_DELTA, LLL_DEF_ETA);
    assert_eq!(st as i32, 0);
    assert!(is_lll_reduced_basis(&b, LLL_DEF_DELTA, LLL_DEF_ETA));
}
