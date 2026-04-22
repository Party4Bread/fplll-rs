//! LLL reduction tests (port of fplll/tests/test_lll.cpp).

use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::integer::Z;
use fplll::io::read_matrix_from_file;
use fplll::lll::{is_lll_reduced_basis, lll_reduce};
use fplll::lll_mpfr::{is_lll_reduced_mpfr, lll_reduce_mpfr};
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;

fn test_lll(mut a: ZMatrix) -> bool {
    let status = lll_reduce(&mut a, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    if status as i32 != 0 {
        eprintln!("LLL failed: {:?}", status);
        return false;
    }
    is_lll_reduced_basis(&a, LLL_DEF_DELTA, LLL_DEF_ETA)
}

#[test]
fn lll_example_in() {
    let a = read_matrix_from_file("tests/lattices/example_in").expect("lattice file");
    assert!(test_lll(a));
}

#[test]
fn lll_dim55_in_mpfr() {
    let mut a = read_matrix_from_file("tests/lattices/dim55_in").expect("lattice file");
    let prec = 128;
    let status = lll_reduce_mpfr(&mut a, prec, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    assert_eq!(status as i32, 0, "LLL_mpfr failed: {:?}", status);
    assert!(is_lll_reduced_mpfr(&a, prec, LLL_DEF_DELTA, LLL_DEF_ETA));
}

#[test]
fn lll_example_in_mpfr() {
    let mut a = read_matrix_from_file("tests/lattices/example_in").expect("lattice file");
    let prec = 64;
    let status = lll_reduce_mpfr(&mut a, prec, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    assert_eq!(status as i32, 0);
    assert!(is_lll_reduced_mpfr(&a, prec, LLL_DEF_DELTA, LLL_DEF_ETA));
}

#[test]
fn lll_intrel_small() {
    let mut rng = RandCtx::new_seeded(42);
    let mut a = ZMatrix::new();
    a.resize(10, 11);
    a.gen_intrel(50, &mut rng);
    assert!(test_lll(a));
}

#[test]
fn lll_intrel_medium() {
    let mut rng = RandCtx::new_seeded(7);
    let mut a = ZMatrix::new();
    a.resize(20, 21);
    a.gen_intrel(200, &mut rng);
    assert!(test_lll(a));
}

#[test]
fn lll_uniform_small() {
    let mut rng = RandCtx::new_seeded(3);
    let mut a = ZMatrix::new();
    a.resize(8, 8);
    a.gen_uniform(30, &mut rng);
    assert!(test_lll(a));
}

#[test]
fn lll_invariant_lattice() {
    // LLL should preserve the lattice: determinant (up to sign) unchanged.
    let mut rng = RandCtx::new_seeded(99);
    let mut a = ZMatrix::new();
    a.resize(6, 6);
    a.gen_uniform(20, &mut rng);
    let det_before = det_of(&a);
    let _ = lll_reduce(&mut a, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    let det_after = det_of(&a);
    // The determinants should match up to sign.
    let mut a2 = det_after.clone(); a2.abs_inplace();
    let mut b2 = det_before.clone(); b2.abs_inplace();
    assert_eq!(a2, b2);
}

/// Bareiss / fraction-free determinant for a square ZMatrix.
fn det_of(a: &ZMatrix) -> Z {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    let mut m: Vec<Vec<Z>> = (0..n).map(|i| (0..n).map(|j| a.get(i, j).clone()).collect()).collect();
    let mut sign: i32 = 1;
    let mut prev = Z::from_i64(1);
    for i in 0..n {
        if m[i][i].is_zero() {
            let mut found = None;
            for k in (i + 1)..n { if !m[k][i].is_zero() { found = Some(k); break; } }
            match found {
                Some(k) => { m.swap(i, k); sign = -sign; }
                None => return Z::zero(),
            }
        }
        for j in (i + 1)..n {
            for k in (i + 1)..n {
                let mut t = Z::zero();
                t.mul_set(&m[i][i], &m[j][k]);
                let mut u = Z::zero();
                u.mul_set(&m[j][i], &m[i][k]);
                t.sub_assign_ref(&u);
                // divide by prev
                let q = Z(t.0.div_exact(&prev.0));
                m[j][k] = q;
            }
        }
        prev = m[i][i].clone();
    }
    let mut out = m[n - 1][n - 1].clone();
    if sign < 0 { out.neg_inplace(); }
    out
}
