//! Babai size-reduction test (port of fplll/tests/test_babai.cpp).
//!
//! Verify that after size reduction by GSO update + row addmul, all
//! |mu[i][j]| <= eta for j < i.

use fplll::defs::LLL_DEF_ETA;
use fplll::gso::{gso_flags, MatGso};
use fplll::lll::{lll_reduce, is_lll_reduced_basis};
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn babai_reduces_mu() {
    let mut rng = RandCtx::new_seeded(55);
    let mut b = ZMatrix::new();
    b.resize(6, 6);
    b.gen_uniform(20, &mut rng);
    let _ = lll_reduce(&mut b, 0.99, LLL_DEF_ETA, 0);
    // After LLL, all mu should be below eta.
    let mut gso = MatGso::new(b.clone(), gso_flags::ROW_EXPO);
    assert!(gso.update_gso());
    for i in 0..gso.d {
        for j in 0..i {
            let (m, e) = gso.get_mu_exp(i, j);
            let true_mu = m * 2f64.powi(e as i32);
            assert!(true_mu.abs() <= LLL_DEF_ETA + 1e-9,
                    "mu[{}][{}] = {} exceeds eta", i, j, true_mu);
        }
    }
    assert!(is_lll_reduced_basis(&b, 0.99, LLL_DEF_ETA));
}
