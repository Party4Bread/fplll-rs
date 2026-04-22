//! Gram-Schmidt orthogonalisation over a lattice basis.
//!
//! This is a simplified port of fplll's `MatGSO<Z_NR<mpz_t>, FP_NR<double>>`.
//! It keeps the `mu` and `r` matrices lazily, and provides the row
//! operations that LLL, BKZ and enumeration depend on.

use crate::float::{frexp, ldexp, rnd_i64};
use crate::integer::Z;
use crate::matrix::ZMatrix;

/// Compute the floating-point value `f * 2^e` equal to the integer `z`, with |f| < 2^53.
#[inline]
fn z_to_f_exp(z: &Z) -> (f64, i64) {
    if z.is_zero() { return (0.0, 0); }
    let (m, e) = z.get_d_2exp();
    // m * 2^e = z. Return (m, e).
    (m, e)
}

/// Flags mirroring MatGSOInterfaceFlags.
pub mod gso_flags {
    pub const DEFAULT: i32 = 0;
    pub const INT_GRAM: i32 = 1;
    pub const ROW_EXPO: i32 = 2;
    pub const OP_FORCE_LONG: i32 = 4;
}

/// Gram-Schmidt state for an integer basis.
pub struct MatGso {
    pub b: ZMatrix,
    /// b_rows, b_cols
    pub d: usize,
    pub n: usize,
    /// Optional transformation matrix u (so that b = u * b_initial). If empty, not tracked.
    pub u: Option<ZMatrix>,
    pub u_inv_t: Option<ZMatrix>,

    /// mu[i][j], lower triangular, j < i
    pub mu: Vec<Vec<f64>>,
    /// r[i][j] = <b_i, b*_j> for j<i, r[i][i] = ||b*_i||^2
    pub r: Vec<Vec<f64>>,

    /// Per-row exponent; row i's "floating" value equals b_i / 2^row_expo[i].
    pub row_expo: Vec<i64>,

    pub enable_row_expo: bool,
    pub enable_int_gram: bool,
    pub row_op_force_long: bool,

    /// Up to which row has been GS-computed.
    pub n_known_rows: usize,
    /// Per-row: how many columns of mu/r are valid.
    pub gso_valid_cols: Vec<usize>,
}

impl MatGso {
    pub fn new(b: ZMatrix, flags: i32) -> Self {
        let d = b.nrows();
        let n = b.ncols();
        let enable_row_expo = (flags & gso_flags::ROW_EXPO) != 0;
        let enable_int_gram = (flags & gso_flags::INT_GRAM) != 0;
        let row_op_force_long = (flags & gso_flags::OP_FORCE_LONG) != 0;
        MatGso {
            b,
            d, n,
            u: None, u_inv_t: None,
            mu: vec![vec![0.0; d]; d],
            r: vec![vec![0.0; d]; d],
            row_expo: vec![0; d],
            enable_row_expo,
            enable_int_gram,
            row_op_force_long,
            n_known_rows: 0,
            gso_valid_cols: vec![0; d],
        }
    }

    pub fn with_transforms(mut self, u: ZMatrix, u_inv_t: ZMatrix) -> Self {
        self.u = Some(u);
        self.u_inv_t = Some(u_inv_t);
        self
    }

    /// Re-evaluate the floating-point exponent for row i.
    fn update_row_expo(&mut self, i: usize) {
        if !self.enable_row_expo { self.row_expo[i] = 0; return; }
        let mut maxe: i64 = i64::MIN;
        for j in 0..self.n {
            let z = self.b.get(i, j);
            if z.is_zero() { continue; }
            let bits = z.0.significant_bits() as i64;
            if bits > maxe { maxe = bits; }
        }
        self.row_expo[i] = if maxe == i64::MIN { 0 } else { maxe };
    }

    /// Compute dot product (b[i], b[j]) as f64 * 2^e with e = row_expo[i] + row_expo[j] (if enabled).
    fn gram(&self, i: usize, j: usize) -> f64 {
        // exact integer dot then convert
        let mut dot = Z::zero();
        for k in 0..self.n {
            dot.addmul(self.b.get(i, k), self.b.get(j, k));
        }
        if dot.is_zero() { return 0.0; }
        let (m, e) = z_to_f_exp(&dot);
        // return value in f64 *2^(e - (row_expo[i]+row_expo[j]))
        let shift = e - (self.row_expo[i] + self.row_expo[j]);
        ldexp(m, shift as i32)
    }

    pub fn invalidate_row(&mut self, i: usize) {
        // Invalidate row i and all downstream rows (they depend on b*_i).
        for k in i..self.d {
            self.gso_valid_cols[k] = 0;
        }
        self.n_known_rows = self.n_known_rows.min(i);
    }

    /// Compute r[i][j], mu[i][j] for j=0..=last_j_max. Ensures rows 0..i are GSO-computed first.
    pub fn update_gso_row(&mut self, i: usize, last_j_max: usize) -> bool {
        if i >= self.d { return false; }

        // Ensure rows 0..i are up to date first.
        for k in 0..i {
            if self.gso_valid_cols[k] <= k {
                if !self.update_gso_row_inner(k, k) { return false; }
            }
        }
        self.update_gso_row_inner(i, last_j_max)
    }

    fn update_gso_row_inner(&mut self, i: usize, last_j_max: usize) -> bool {
        if i >= self.d { return false; }

        // Ensure row_expo set if in use.
        if self.enable_row_expo { self.update_row_expo(i); }
        // For rows <= i in row_expo, also.
        if self.enable_row_expo {
            for k in 0..i { if self.gso_valid_cols[k] == 0 { self.update_row_expo(k); } }
        }

        let start = self.gso_valid_cols[i];
        let last = last_j_max.min(i);
        for j in start..=last {
            let mut s = self.gram(i, j);
            if !s.is_finite() { return false; }
            for k in 0..j {
                s -= self.mu[j][k] * self.r[i][k];
            }
            self.r[i][j] = s;
            if j < i {
                let rjj = self.r[j][j];
                if rjj == 0.0 { return false; }
                self.mu[i][j] = s / rjj;
            }
        }
        self.gso_valid_cols[i] = last + 1;
        if i >= self.n_known_rows { self.n_known_rows = i + 1; }
        true
    }

    /// Full GSO recompute up through row `d-1`.
    pub fn update_gso(&mut self) -> bool {
        for i in 0..self.d {
            if !self.update_gso_row(i, i) { return false; }
        }
        true
    }

    #[inline] pub fn get_mu(&self, i: usize, j: usize) -> f64 { self.mu[i][j] }
    #[inline] pub fn get_r(&self, i: usize, j: usize) -> f64 { self.r[i][j] }

    /// r(i,j) as f64 times 2^(row_expo[i] + row_expo[j]) when ROW_EXPO is on.
    /// In this port r is already in the row-expo scaled domain, so this is `r[i][j]` paired with expo.
    pub fn get_r_exp(&self, i: usize, j: usize) -> (f64, i64) {
        let e = if self.enable_row_expo { self.row_expo[i] + self.row_expo[j] } else { 0 };
        (self.r[i][j], e)
    }
    pub fn get_mu_exp(&self, i: usize, j: usize) -> (f64, i64) {
        let e = if self.enable_row_expo { self.row_expo[i] - self.row_expo[j] } else { 0 };
        (self.mu[i][j], e)
    }

    /// Apply b[i] -= x * b[j] (where `x` is an FP rounded to an integer), i > j.
    /// Returns true on success.
    pub fn row_addmul(&mut self, i: usize, j: usize, x_f: f64) {
        if x_f == 0.0 { return; }
        let xi = rnd_i64(x_f);
        if xi == 0 { return; }
        // b[i] -= xi * b[j]
        // Use wide integer: create Z from xi.
        let z = Z::from_i64(-xi);
        self.b.row_addmul_vec(i, j, &z, self.n);
        if let Some(u) = self.u.as_mut() {
            u.row_addmul_vec(i, j, &z, u.ncols());
        }
        if let Some(uinv) = self.u_inv_t.as_mut() {
            let nz = Z::from_i64(xi);
            uinv.row_addmul_vec(j, i, &nz, uinv.ncols());
        }
        // Update mu row i approximately: mu[i][k] -= xi * mu[j][k] for k < j
        for k in 0..j { self.mu[i][k] -= (xi as f64) * self.mu[j][k]; }
        // mu[i][j] is invalidated (becomes residual)
        self.invalidate_row(i);
    }

    /// Apply b[i] -= (x * 2^e) * b[j] with integer x.
    pub fn row_addmul_2si(&mut self, i: usize, j: usize, x: i64, e: i64) {
        if x == 0 { return; }
        let mut xz = Z::from_i64(x);
        xz.neg_inplace();
        self.b.row_addmul_2si_vec(i, j, &xz, e, self.n);
        if let Some(u) = self.u.as_mut() {
            u.row_addmul_2si_vec(i, j, &xz, e, u.ncols());
        }
        self.invalidate_row(i);
    }

    /// Exchange rows (i, i+1) and mark affected rows invalid.
    pub fn move_row(&mut self, old: usize, new: usize) {
        if old == new { return; }
        // Rotate
        if old < new {
            for k in old..new { self.b.swap_rows(k, k + 1); }
            if let Some(u) = self.u.as_mut() {
                for k in old..new { u.swap_rows(k, k + 1); }
            }
        } else {
            let mut k = old;
            while k > new { self.b.swap_rows(k, k - 1); k -= 1; }
            if let Some(u) = self.u.as_mut() {
                let mut k = old;
                while k > new { u.swap_rows(k, k - 1); k -= 1; }
            }
        }
        let lo = old.min(new);
        for k in lo..self.d {
            self.gso_valid_cols[k] = 0;
        }
        self.n_known_rows = self.n_known_rows.min(lo);
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j { return; }
        self.b.swap_rows(i, j);
        if let Some(u) = self.u.as_mut() { u.swap_rows(i, j); }
        let lo = i.min(j);
        for k in lo..self.d { self.gso_valid_cols[k] = 0; }
        self.n_known_rows = self.n_known_rows.min(lo);
    }

    pub fn get_max_exp_of_b(&self) -> i64 { self.b.max_exp_of_b() }

    pub fn b_row_is_zero(&self, i: usize) -> bool { self.b.b_row_is_zero(i) }

    /// Largest exponent of |mu[i][j]| for j < n_columns (approx).
    pub fn get_max_mu_exp(&self, i: usize, n_columns: usize) -> i64 {
        let mut m: i64 = i64::MIN;
        for j in 0..n_columns.min(i) {
            let v = self.mu[i][j].abs();
            if v == 0.0 { continue; }
            let (_, e) = frexp(v);
            if (e as i64) > m { m = e as i64; }
        }
        if m == i64::MIN { 0 } else { m }
    }

    pub fn set_r(&mut self, i: usize, j: usize, v: f64) { self.r[i][j] = v; }

    /// Returns r[i][i] (||b*_i||^2).
    pub fn r_diag(&self, i: usize) -> f64 { self.r[i][i] }
}
