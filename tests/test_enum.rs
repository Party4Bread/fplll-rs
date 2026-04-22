//! Enumeration tests.

use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::enumerate::{enumerate_svp, Enumeration, FastEvaluator};
use fplll::gso::MatGso;
use fplll::lll::lll_reduce;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn enum_trivial_identity() {
    let mut m = ZMatrix::new();
    m.gen_identity(4);
    let mut gso = MatGso::new(m, 0);
    assert!(gso.update_gso());
    // Any unit vector has length 1; no shorter exists, so set bound 2.
    let ev = enumerate_svp(&gso, 2.0, 1);
    assert!(!ev.solutions.is_empty());
    let (d, _x) = ev.solutions[0].clone();
    assert!(d >= 1.0 - 1e-12);
    assert!(d <= 2.0 + 1e-12);
}

#[test]
fn enum_random_basis_after_lll() {
    let mut rng = RandCtx::new_seeded(321);
    let mut b = ZMatrix::new();
    b.resize(6, 6);
    b.gen_uniform(15, &mut rng);
    let _ = lll_reduce(&mut b, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    let mut gso = MatGso::new(b.clone(), 0);
    assert!(gso.update_gso());
    // Use ||b*_0||^2 as bound.
    let r0 = gso.r_diag(0);
    let mut ev = FastEvaluator::new();
    let mut en = Enumeration::new(&gso);
    en.enumerate(0, gso.d, r0 * 2.0, None, None, &mut ev);
    assert!(!ev.solutions.is_empty());
}
