//! Pruner tests (port of fplll/tests/test_pruner.cpp subset).

use fplll::pruner::{enumeration_cost, linear_pruning, no_pruning, step_pruning};

#[test]
fn linear_pruning_shape() {
    let p = linear_pruning(10);
    assert_eq!(p.len(), 10);
    assert_eq!(p[0], 0.0);
    assert!(p[p.len() - 1] >= 1.0 - 1e-12);
    for i in 1..p.len() { assert!(p[i] >= p[i - 1]); }
}

#[test]
fn no_pruning_is_flat() {
    let p = no_pruning(6);
    assert!(p.iter().all(|x| (*x - 1.0).abs() < 1e-12));
}

#[test]
fn step_pruning_keeps_top() {
    let p = step_pruning(10, 3, 0.5);
    assert_eq!(p[9], 1.0);
    assert_eq!(p[7], 1.0);
    assert_eq!(p[6], 0.5);
    assert_eq!(p[0], 0.5);
}

#[test]
fn enumeration_cost_finite() {
    let r: Vec<f64> = (0..20).map(|i| (20 - i) as f64 * 100.0).collect();
    let c = enumeration_cost(&r, 10_000.0);
    assert!(c.is_finite());
    assert!(c > 0.0);
}
