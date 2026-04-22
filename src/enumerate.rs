//! Lattice enumeration (Schnorr–Euchner) over a GSO for SVP / CVP.
//!
//! Port of fplll/enum/enumerate.cpp (FT = double) without the
//! recursive-specialised dispatcher; the main loop is the equivalent of
//! fplll's `enumerate_loop()`.

use crate::gso::MatGso;

/// Evaluator: collects the `max_sols` best solutions, shortest first.
pub struct FastEvaluator {
    pub solutions: Vec<(f64, Vec<i64>)>,
    pub max_sols: usize,
    pub sub_solutions: Vec<(f64, Vec<i64>)>,
    pub find_sub: bool,
}

impl FastEvaluator {
    pub fn new() -> Self { FastEvaluator { solutions: Vec::new(), max_sols: 1, sub_solutions: Vec::new(), find_sub: false } }
    pub fn with_max_sols(max: usize) -> Self { FastEvaluator { solutions: Vec::new(), max_sols: max, sub_solutions: Vec::new(), find_sub: false } }

    pub fn insert(&mut self, dist: f64, x: Vec<i64>) {
        let pos = self.solutions.partition_point(|(d, _)| *d < dist);
        self.solutions.insert(pos, (dist, x));
        if self.solutions.len() > self.max_sols {
            self.solutions.truncate(self.max_sols);
        }
    }
    pub fn best(&self) -> Option<&(f64, Vec<i64>)> { self.solutions.first() }
}

pub struct Enumeration<'a> {
    gso: &'a MatGso,
    pub nodes: u64,
}

impl<'a> Enumeration<'a> {
    pub fn new(gso: &'a MatGso) -> Self { Enumeration { gso, nodes: 0 } }

    /// Enumerate lattice vectors with squared length <= `max_dist`.
    pub fn enumerate(
        &mut self,
        first: usize, last: usize,
        max_dist: f64,
        target_coord: Option<&[f64]>,
        pruning: Option<&[f64]>,
        eval: &mut FastEvaluator,
    ) -> f64 {
        let d = last - first;
        if d == 0 { return max_dist; }

        let mut mut_: Vec<Vec<f64>> = vec![vec![0.0; d]; d];
        let mut rdiag: Vec<f64> = vec![0.0; d];
        for i in 0..d {
            rdiag[i] = self.gso.r_diag(first + i);
            for j in (i + 1)..d {
                mut_[i][j] = self.gso.get_mu(first + j, first + i);
            }
        }

        let mut maxdist = max_dist;
        let mut pbounds = vec![maxdist; d];
        if let Some(p) = pruning {
            for i in 0..d {
                let pr = *p.get(i).unwrap_or(&1.0);
                pbounds[i] = maxdist * pr;
            }
        }

        let is_svp = target_coord.is_none();
        let tgt: Vec<f64> = match target_coord {
            Some(t) => t[..d].to_vec(),
            None => vec![0.0; d],
        };

        let mut x: Vec<i64> = vec![0; d];
        let mut dx: Vec<i64> = vec![0; d];
        let mut ddx: Vec<i64> = vec![0; d];
        let mut center: Vec<f64> = vec![0.0; d];
        let mut partdist: Vec<f64> = vec![0.0; d + 1];
        let mut center_partsum: Vec<Vec<f64>> = vec![vec![0.0; d + 1]; d];
        for i in 0..d { center_partsum[i][d] = tgt[i]; }

        let mut k = d - 1;
        center[k] = center_partsum[k][d];
        x[k] = center[k].round() as i64;
        dx[k] = if center[k] >= x[k] as f64 { 1 } else { -1 };
        ddx[k] = dx[k];
        partdist[d] = 0.0;

        let mut finished = false;
        while !finished {
            let ak = x[k] as f64 - center[k];
            let newdist = partdist[k + 1] + ak * ak * rdiag[k];
            if newdist <= pbounds[k] {
                self.nodes += 1;
                partdist[k] = newdist;
                if k == 0 {
                    if !is_svp || newdist > 0.0 {
                        eval.insert(newdist, x.clone());
                        if eval.solutions.len() >= eval.max_sols {
                            if let Some((d0, _)) = eval.solutions.last() {
                                maxdist = *d0;
                                for b in pbounds.iter_mut() { *b = b.min(maxdist); }
                            }
                        }
                    }
                    // At k=0, always zigzag the sibling and re-evaluate in the next iteration.
                    x[0] += dx[0];
                    ddx[0] = -ddx[0];
                    dx[0] = ddx[0] - dx[0];
                } else {
                    // Descend: compute center[k-1] from partial sums of x at levels above.
                    for j in (k..d).rev() {
                        let prev = if j == d - 1 { tgt[k - 1] } else { center_partsum[k - 1][j + 1] };
                        center_partsum[k - 1][j] = prev - (x[j] as f64) * mut_[k - 1][j];
                    }
                    let newcenter = center_partsum[k - 1][k];
                    center[k - 1] = newcenter;
                    k -= 1;
                    x[k] = newcenter.round() as i64;
                    dx[k] = if newcenter >= x[k] as f64 { 1 } else { -1 };
                    ddx[k] = dx[k];
                }
            } else {
                // Out of bound: go up and try next sibling there.
                if !next_pos_up(&mut k, d, is_svp, &mut x, &mut dx, &mut ddx, &partdist) {
                    finished = true;
                }
            }
        }

        if let Some((d0, _)) = eval.solutions.first() { *d0 } else { maxdist }
    }
}

fn next_pos_up(
    k: &mut usize, d: usize, is_svp: bool,
    x: &mut [i64], dx: &mut [i64], ddx: &mut [i64],
    partdist: &[f64],
) -> bool {
    *k += 1;
    if *k >= d { return false; }
    if partdist[*k + 1] != 0.0 {
        x[*k] += dx[*k];
        ddx[*k] = -ddx[*k];
        dx[*k] = ddx[*k] - dx[*k];
    } else if is_svp {
        x[*k] += 1;
    } else {
        x[*k] += dx[*k];
        ddx[*k] = -ddx[*k];
        dx[*k] = ddx[*k] - dx[*k];
    }
    true
}

pub fn enumerate_svp(gso: &MatGso, max_dist: f64, max_sols: usize) -> FastEvaluator {
    let mut eval = FastEvaluator::with_max_sols(max_sols);
    let mut en = Enumeration::new(gso);
    en.enumerate(0, gso.d, max_dist, None, None, &mut eval);
    eval
}

pub fn enumerate_cvp(gso: &MatGso, max_dist: f64, target_coord: &[f64]) -> FastEvaluator {
    let mut eval = FastEvaluator::new();
    let mut en = Enumeration::new(gso);
    en.enumerate(0, gso.d, max_dist, Some(target_coord), None, &mut eval);
    eval
}
