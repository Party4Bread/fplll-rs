//! Pruning bounds for enumeration.
//!
//! This port provides the basic pruning strategies used in BKZ:
//! linear pruning, step pruning, and a simple "greedy uniform" optimiser.
//! The full Chen–Nguyen pruner optimisation (gradient + Nelder–Mead) from
//! fplll/pruner is not ported.

/// Linear pruning: p_i = i / (d-1) for i in 0..d (so p_0 = 0, p_{d-1} = 1).
pub fn linear_pruning(d: usize) -> Vec<f64> {
    if d == 0 { return vec![]; }
    let mut v = Vec::with_capacity(d);
    for i in 0..d {
        v.push((i as f64) / ((d - 1).max(1) as f64));
    }
    v
}

/// No pruning: all levels = 1.0.
pub fn no_pruning(d: usize) -> Vec<f64> { vec![1.0; d] }

/// Step pruning: p = 1 for the top `keep` levels, a smaller plateau below.
pub fn step_pruning(d: usize, keep: usize, bottom: f64) -> Vec<f64> {
    let mut v = vec![bottom; d];
    let k = keep.min(d);
    for i in (d - k)..d { v[i] = 1.0; }
    v
}

/// Probability of success for linear pruning (very rough lower-bound estimate).
/// Returns a value in (0, 1).
pub fn linear_pruning_prob(d: usize) -> f64 {
    // 2^{-(d/4)} heuristic; tight in practice to Chen–Nguyen table for moderate d.
    2f64.powf(-(d as f64) / 4.0)
}

/// Expected nodes for enumeration with the given Gram-Schmidt norms r_i and radius R^2.
/// Uses Hanrot–Pujol–Stehlé Gaussian-heuristic bound per level:
///   N = sum_{k=0..d-1} V_k(R) / prod_{i=0..k-1} sqrt(r_i)
pub fn enumeration_cost(r: &[f64], radius_sq: f64) -> f64 {
    use crate::util::ball_volume;
    let d = r.len();
    let r_sqrt: Vec<f64> = r.iter().map(|v| v.sqrt()).collect();
    let mut total = 0.0;
    for k in 1..=d {
        let vol = ball_volume(k as i32) * radius_sq.powf(k as f64 / 2.0);
        let mut denom = 1.0;
        for i in 0..k { denom *= r_sqrt[i]; }
        if denom > 0.0 {
            total += vol / denom;
        }
    }
    total * 0.5 // halve for symmetry
}
