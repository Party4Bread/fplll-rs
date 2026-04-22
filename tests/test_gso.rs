//! GSO tests (port of fplll/tests/test_gso.cpp subset).

use fplll::gso::{gso_flags, MatGso};
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn gso_identity() {
    let mut m = ZMatrix::new();
    m.gen_identity(5);
    let mut gso = MatGso::new(m, 0);
    assert!(gso.update_gso());
    for i in 0..5 {
        for j in 0..i { assert_eq!(gso.get_mu(i, j), 0.0); }
        assert_eq!(gso.r_diag(i), 1.0);
    }
}

#[test]
fn gso_triangular() {
    let mut rng = RandCtx::new_seeded(1);
    let mut m = ZMatrix::new();
    m.resize(4, 4);
    m.gen_trg(1.2, &mut rng);
    let mut gso = MatGso::new(m, 0);
    assert!(gso.update_gso());
    // All r_diag > 0
    for i in 0..4 { assert!(gso.r_diag(i) > 0.0); }
}

#[test]
fn gso_row_expo_consistency() {
    let mut rng = RandCtx::new_seeded(2);
    let mut m = ZMatrix::new();
    m.resize(6, 6);
    m.gen_uniform(40, &mut rng);
    let m2 = m.clone();
    let mut gso_a = MatGso::new(m, 0);
    let mut gso_b = MatGso::new(m2, gso_flags::ROW_EXPO);
    assert!(gso_a.update_gso());
    assert!(gso_b.update_gso());
    for i in 0..gso_a.d {
        let ra = gso_a.r_diag(i);
        let (rb, eb) = gso_b.get_r_exp(i, i);
        let rb_unscaled = rb * 2f64.powi(eb as i32);
        assert!((ra - rb_unscaled).abs() / ra.abs() < 1e-8, "mismatch at {}: {} vs {}", i, ra, rb_unscaled);
    }
}
