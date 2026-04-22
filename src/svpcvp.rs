//! High-level SVP/CVP entry points: given a basis, find a shortest vector
//! or a closest vector to a target.

use crate::defs::{LLL_DEF_DELTA, LLL_DEF_ETA, SvpMethod};
use crate::enumerate::{Enumeration, FastEvaluator};
use crate::gso::MatGso;
use crate::integer::Z;
use crate::lll::lll_reduce;
use crate::matrix::ZMatrix;
use crate::pruner::linear_pruning;

/// Returns coordinates of a shortest vector in `b` (as the coefficients in the given basis).
pub fn shortest_vector(b: &mut ZMatrix, _method: SvpMethod, _flags: i32) -> Result<Vec<Z>, &'static str> {
    // First ensure LLL-reduced.
    let _ = lll_reduce(b, LLL_DEF_DELTA, LLL_DEF_ETA, 0);

    let taken = std::mem::take(b);
    let mut gso = MatGso::new(taken, 0);
    if !gso.update_gso() { *b = gso.b; return Err("GSO failure"); }

    let d = gso.d;
    // Initial radius: first basis vector squared norm.
    let mut max_dist = 0.0;
    for j in 0..gso.n {
        let v = gso.b.get(0, j).get_d();
        max_dist += v * v;
    }

    let mut eval = FastEvaluator::new();
    let mut en = Enumeration::new(&gso);
    en.enumerate(0, d, max_dist, None, None, &mut eval);
    if eval.solutions.is_empty() { *b = gso.b; return Err("no vector found"); }
    let x = eval.solutions.remove(0).1;
    let coord: Vec<Z> = x.into_iter().map(Z::from_i64).collect();
    *b = gso.b;
    Ok(coord)
}

/// SVP with caller-provided linear pruning radii (one value per level, in [0,1]).
pub fn shortest_vector_pruning(b: &mut ZMatrix, pruning: &[f64], _flags: i32) -> Result<Vec<Z>, &'static str> {
    let _ = lll_reduce(b, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    let taken = std::mem::take(b);
    let mut gso = MatGso::new(taken, 0);
    if !gso.update_gso() { *b = gso.b; return Err("GSO failure"); }

    let d = gso.d;
    let mut max_dist = 0.0;
    for j in 0..gso.n {
        let v = gso.b.get(0, j).get_d();
        max_dist += v * v;
    }

    let effective_pruning = if pruning.len() == d { pruning.to_vec() } else { linear_pruning(d) };

    let mut eval = FastEvaluator::new();
    let mut en = Enumeration::new(&gso);
    en.enumerate(0, d, max_dist, None, Some(&effective_pruning), &mut eval);
    if eval.solutions.is_empty() { *b = gso.b; return Err("no vector found"); }
    let x = eval.solutions.remove(0).1;
    let coord: Vec<Z> = x.into_iter().map(Z::from_i64).collect();
    *b = gso.b;
    Ok(coord)
}

/// Find a closest lattice vector to `target` and return it.
pub fn closest_vector(b: &mut ZMatrix, target: &[Z]) -> Result<Vec<Z>, &'static str> {
    let _ = lll_reduce(b, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
    let taken = std::mem::take(b);
    let mut gso = MatGso::new(taken, 0);
    if !gso.update_gso() { *b = gso.b; return Err("GSO failure"); }

    let d = gso.d;

    // Translate target into GSO coordinates: t_i = <target, b*_i> / <b*_i, b*_i>.
    // Since we only have mu/r, compute: tproj_i = <target, b_i> (integer) projected.
    // Actually, the enumeration expects center_partsum[i][d] = target coord in GS basis.
    // We'll compute target GS coordinates by recursion.
    let mut t_coord = vec![0.0; d];
    for i in 0..d {
        // <target, b_i>
        let mut dot = 0.0;
        for j in 0..gso.n {
            let bv = gso.b.get(i, j).get_d();
            let tv = target[j].get_d();
            dot += bv * tv;
        }
        let mut s = dot;
        for k in 0..i { s -= gso.get_mu(i, k) * t_coord[k] * gso.r_diag(k); }
        if gso.r_diag(i) == 0.0 { return Err("singular GSO"); }
        t_coord[i] = s / gso.r_diag(i);
    }

    // Initial radius: large; use ||target - b_0||^2 as a starting bound (not necessarily tight but finite).
    let mut max_dist = 0.0;
    for j in 0..gso.n {
        let d_ij = gso.b.get(0, j).get_d() - target[j].get_d();
        max_dist += d_ij * d_ij;
    }

    let mut eval = FastEvaluator::new();
    let mut en = Enumeration::new(&gso);
    en.enumerate(0, d, max_dist, Some(&t_coord), None, &mut eval);

    if eval.solutions.is_empty() { *b = gso.b; return Err("no vector found"); }
    let x = eval.solutions.remove(0).1;

    // Reconstruct vector = sum x_i * b_i
    let mut result = vec![Z::zero(); gso.n];
    for (i, xi) in x.iter().enumerate() {
        if *xi == 0 { continue; }
        let xz = Z::from_i64(*xi);
        for j in 0..gso.n {
            result[j].addmul(&xz, gso.b.get(i, j));
        }
    }
    *b = gso.b;
    Ok(result)
}
