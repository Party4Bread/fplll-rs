//! Lattice generator tests.

use fplll::integer::Z;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

#[test]
fn gen_identity_is_identity() {
    let mut m = ZMatrix::new();
    m.gen_identity(5);
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(*m.get(i, j), if i == j { Z::one() } else { Z::zero() });
        }
    }
}

#[test]
fn gen_uniform_has_bounded_entries() {
    let mut rng = RandCtx::new_seeded(1);
    let mut m = ZMatrix::new();
    m.resize(5, 5);
    m.gen_uniform(10, &mut rng);
    for i in 0..5 {
        for j in 0..5 {
            let v = m.get(i, j).get_si().abs();
            assert!(v < (1 << 10));
        }
    }
}

#[test]
fn gen_intrel_has_identity_tail() {
    let mut rng = RandCtx::new_seeded(2);
    let mut m = ZMatrix::new();
    m.resize(6, 7);
    m.gen_intrel(30, &mut rng);
    // column 1..7 should be identity shifted.
    for i in 0..6 {
        for j in 1..7 {
            let expected = if i + 1 == j { Z::one() } else { Z::zero() };
            assert_eq!(*m.get(i, j), expected);
        }
    }
}

#[test]
fn gen_ntrulike_block_structure() {
    let mut rng = RandCtx::new_seeded(3);
    let d = 4;
    let mut m = ZMatrix::new();
    m.resize(2 * d, 2 * d);
    let q = Z::from_i64(97);
    m.gen_ntrulike(&q, &mut rng);
    // Top-left block is identity
    for i in 0..d {
        for j in 0..d {
            let e = if i == j { Z::one() } else { Z::zero() };
            assert_eq!(*m.get(i, j), e);
        }
    }
    // Bottom-left is zero
    for i in 0..d {
        for j in 0..d {
            assert!(m.get(d + i, j).is_zero());
        }
    }
    // Bottom-right is q*I
    for i in 0..d {
        for j in 0..d {
            let e = if i == j { q.clone() } else { Z::zero() };
            assert_eq!(*m.get(d + i, d + j), e);
        }
    }
}

#[test]
fn gen_qary_structure() {
    let mut rng = RandCtx::new_seeded(4);
    let mut m = ZMatrix::new();
    m.resize(6, 6);
    let q = Z::from_i64(101);
    m.gen_qary(2, &q, &mut rng);
    // Top-left block (n-k=4, n-k=4) is identity
    for i in 0..4 {
        for j in 0..4 {
            let e = if i == j { Z::one() } else { Z::zero() };
            assert_eq!(*m.get(i, j), e);
        }
    }
}
