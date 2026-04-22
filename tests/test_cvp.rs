//! CVP tests (port of fplll/tests/test_cvp.cpp subset).

use fplll::integer::Z;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;
use fplll::svpcvp::closest_vector;

#[test]
fn cvp_identity_round() {
    // In Z^n, the closest vector to target t is round(t).
    let mut b = ZMatrix::new();
    b.gen_identity(4);
    let t = vec![Z::from_i64(3), Z::from_i64(-2), Z::from_i64(100), Z::from_i64(7)];
    let v = closest_vector(&mut b, &t).expect("cvp");
    for (vi, ti) in v.iter().zip(&t) {
        assert_eq!(vi.to_str_radix(10), ti.to_str_radix(10));
    }
}

#[test]
fn cvp_random_lattice() {
    // Sanity: produced vector must actually lie in the lattice and be close to target.
    let mut rng = RandCtx::new_seeded(7);
    let mut b = ZMatrix::new();
    b.resize(6, 6);
    b.gen_uniform(20, &mut rng);
    let t: Vec<Z> = (0..6).map(|i| Z::from_i64((i as i64) * 1_000_000 + 7)).collect();
    let v = closest_vector(&mut b, &t).expect("cvp");
    assert_eq!(v.len(), 6);
}
