//! Rounding / ceiling sanity checks (port of fplll/tests/test_ceil.cpp).

use fplll::float::{rnd, rnd_i64};

#[test]
fn basic_rounding() {
    assert_eq!(rnd_i64(0.0), 0);
    assert_eq!(rnd_i64(0.499), 0);
    assert_eq!(rnd_i64(-0.499), 0);
    assert_eq!(rnd_i64(0.501), 1);
    assert_eq!(rnd_i64(-0.501), -1);
    assert_eq!(rnd_i64(2.5), 2); // tie to even
    assert_eq!(rnd_i64(3.5), 4);
}

#[test]
fn rnd_f64() {
    assert_eq!(rnd(1.0), 1.0);
    assert_eq!(rnd(-1.0), -1.0);
    assert_eq!(rnd(0.5), 0.0); // tie to even
    assert_eq!(rnd(1.5), 2.0);
}
