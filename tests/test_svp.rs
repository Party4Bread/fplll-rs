//! SVP tests (port of fplll/tests/test_svp.cpp subset).

use fplll::defs::SvpMethod;
use fplll::integer::Z;
use fplll::io::read_matrix_from_file;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;
use fplll::svpcvp::shortest_vector;

fn sqnorm_of_combination(b: &ZMatrix, coord: &[Z]) -> f64 {
    let n = b.ncols();
    let mut v = vec![Z::zero(); n];
    for (i, c) in coord.iter().enumerate() {
        for j in 0..n { v[j].addmul(c, b.get(i, j)); }
    }
    let mut sq = 0.0;
    for x in &v {
        let d = x.get_d();
        sq += d * d;
    }
    sq
}

#[test]
fn svp_small_identity_family() {
    let mut b = ZMatrix::new();
    b.gen_identity(5);
    let coord = shortest_vector(&mut b, SvpMethod::Fast, 0).expect("svp");
    let sq = sqnorm_of_combination(&b, &coord);
    assert_eq!(sq, 1.0); // shortest vector of Z^5 is a unit vector
}

#[test]
fn svp_intrel() {
    let mut rng = RandCtx::new_seeded(101);
    let mut b = ZMatrix::new();
    b.resize(8, 9);
    b.gen_intrel(20, &mut rng);
    let coord = shortest_vector(&mut b, SvpMethod::Fast, 0).expect("svp");
    assert!(!coord.iter().all(|c| c.is_zero()));
}

#[test]
#[ignore = "large SVP test, slow"]
fn svp_example_in() {
    let mut b = read_matrix_from_file("tests/lattices/example_svp_in").expect("lattice file");
    let coord = shortest_vector(&mut b, SvpMethod::Fast, 0).expect("svp");
    let sq = sqnorm_of_combination(&b, &coord);
    assert!(sq > 0.0);
}
