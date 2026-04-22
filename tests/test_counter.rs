//! Counter/node-count test (port of fplll/tests/test_counter.cpp).

use fplll::enumerate::{Enumeration, FastEvaluator};
use fplll::gso::MatGso;
use fplll::matrix::ZMatrix;

#[test]
fn counter_nonzero_after_enum() {
    let mut m = ZMatrix::new();
    m.gen_identity(5);
    let mut gso = MatGso::new(m, 0);
    assert!(gso.update_gso());
    let mut ev = FastEvaluator::new();
    let mut en = Enumeration::new(&gso);
    en.enumerate(0, gso.d, 3.0, None, None, &mut ev);
    assert!(en.nodes > 0);
}
