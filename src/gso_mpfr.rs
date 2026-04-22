//! MPFR-backed Gram-Schmidt orthogonalisation.
//!
//! A parallel implementation of `MatGso` that uses `rug::Float` for the
//! `r` and `mu` matrices. Use this when `f64` precision is not sufficient
//! (e.g. high-dimensional lattices with large entries).

use crate::float_mpfr::{self as fp};
use crate::integer::Z;
use crate::matrix::ZMatrix;
use rug::{Assign, Float};

pub struct MatGsoMpfr {
    pub b: ZMatrix,
    pub d: usize,
    pub n: usize,
    pub prec: u32,

    /// mu[i][j] for j < i, in rug::Float at precision `prec`.
    pub mu: Vec<Vec<Float>>,
    /// r[i][j], r[i][i] = ||b*_i||^2.
    pub r: Vec<Vec<Float>>,

    pub n_known_rows: usize,
    pub gso_valid_cols: Vec<usize>,
}

impl MatGsoMpfr {
    pub fn new(b: ZMatrix, prec: u32) -> Self {
        let d = b.nrows();
        let n = b.ncols();
        let _old = fp::set_prec(prec);
        let mu = (0..d).map(|_| (0..d).map(|_| fp::zero()).collect()).collect();
        let r = (0..d).map(|_| (0..d).map(|_| fp::zero()).collect()).collect();
        MatGsoMpfr {
            b, d, n, prec,
            mu, r,
            n_known_rows: 0,
            gso_valid_cols: vec![0; d],
        }
    }

    /// <b_i, b_j> as Float at precision `prec`. Exact integer dot followed by conversion.
    fn gram(&self, i: usize, j: usize) -> Float {
        let mut dot = Z::zero();
        for k in 0..self.n {
            dot.addmul(self.b.get(i, k), self.b.get(j, k));
        }
        Float::with_val(self.prec, &dot.0)
    }

    pub fn invalidate_row(&mut self, i: usize) {
        for k in i..self.d { self.gso_valid_cols[k] = 0; }
        self.n_known_rows = self.n_known_rows.min(i);
    }

    pub fn update_gso_row(&mut self, i: usize, last_j_max: usize) -> bool {
        if i >= self.d { return false; }
        // Ensure upstream.
        for k in 0..i {
            if self.gso_valid_cols[k] <= k {
                if !self.update_gso_row_inner(k, k) { return false; }
            }
        }
        self.update_gso_row_inner(i, last_j_max)
    }

    fn update_gso_row_inner(&mut self, i: usize, last_j_max: usize) -> bool {
        let _old = fp::set_prec(self.prec);
        let start = self.gso_valid_cols[i];
        let last = last_j_max.min(i);
        for j in start..=last {
            let mut s = self.gram(i, j);
            for k in 0..j {
                // s -= mu[j][k] * r[i][k]
                let prod = Float::with_val(self.prec, &self.mu[j][k] * &self.r[i][k]);
                s -= prod;
            }
            self.r[i][j].assign(&s);
            if j < i {
                if self.r[j][j].is_zero() { return false; }
                self.mu[i][j].assign(&s / &self.r[j][j]);
            }
        }
        self.gso_valid_cols[i] = last + 1;
        if i >= self.n_known_rows { self.n_known_rows = i + 1; }
        true
    }

    pub fn update_gso(&mut self) -> bool {
        for i in 0..self.d {
            if !self.update_gso_row(i, i) { return false; }
        }
        true
    }

    #[inline] pub fn get_mu(&self, i: usize, j: usize) -> &Float { &self.mu[i][j] }
    #[inline] pub fn r_diag(&self, i: usize) -> &Float { &self.r[i][i] }

    /// b[i] -= q * b[j], with q an arbitrary-precision integer (from rounding true mu).
    pub fn row_addmul_z(&mut self, i: usize, j: usize, q: &rug::Integer) {
        if q == &0 { return; }
        // b[i] -= q * b[j]
        let mut neg = Z(q.clone());
        neg.neg_inplace();
        self.b.row_addmul_vec(i, j, &neg, self.n);
        self.invalidate_row(i);
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j { return; }
        self.b.swap_rows(i, j);
        let lo = i.min(j);
        for k in lo..self.d { self.gso_valid_cols[k] = 0; }
        self.n_known_rows = self.n_known_rows.min(lo);
    }

    pub fn move_row(&mut self, old: usize, new: usize) {
        if old == new { return; }
        if old < new {
            for k in old..new { self.b.swap_rows(k, k + 1); }
        } else {
            let mut k = old;
            while k > new { self.b.swap_rows(k, k - 1); k -= 1; }
        }
        let lo = old.min(new);
        for k in lo..self.d { self.gso_valid_cols[k] = 0; }
        self.n_known_rows = self.n_known_rows.min(lo);
    }

    pub fn b_row_is_zero(&self, i: usize) -> bool { self.b.b_row_is_zero(i) }
    pub fn get_max_exp_of_b(&self) -> i64 { self.b.max_exp_of_b() }
}
