//! LLL lattice reduction.
//!
//! Follows the classical Lenstra–Lenstra–Lovász algorithm, aligning with
//! fplll/lll.cpp. Works on a `MatGso` whose basis is the matrix of row
//! vectors to reduce.

use crate::defs::{LLL_DEF_DELTA, LLL_DEF_ETA, RedStatus};
use crate::float::frexp;
use crate::gso::MatGso;
use crate::matrix::ZMatrix;

/// Round-with-exponent: return (mantissa, expo) such that
/// `mantissa * 2^expo ~= round(stored * 2^native_expo)`, with `mantissa` fitting in i64.
fn round_we(stored: f64, native_expo: i64) -> (i64, i64) {
    if stored == 0.0 { return (0, 0); }
    let true_val = stored * 2f64.powi(native_expo as i32);
    if true_val.abs() < 9.2e18 {
        // Fits in i64.
        return (true_val.round() as i64, 0);
    }
    // Split: keep 62 bits of mantissa.
    let (m, e) = frexp(true_val); // true_val = m * 2^e, m in [0.5, 1)
    let shift = e - 62;
    if shift > 0 {
        // mantissa = round(m * 2^62), exp = shift
        let mantissa = (m * (1i64 << 62) as f64).round() as i64;
        (mantissa, shift as i64)
    } else {
        ((true_val.round() as i64).max(i64::MIN).min(i64::MAX), 0)
    }
}

pub struct LllReduction<'a> {
    gso: &'a mut MatGso,
    pub delta: f64,
    pub eta: f64,
    pub swap_threshold: f64,
    pub siegel: bool,
    pub early_red: bool,
    pub verbose: bool,
    pub status: RedStatus,
    pub final_kappa: usize,
    pub zeros: usize,
    pub n_swaps: u64,
}

impl<'a> LllReduction<'a> {
    pub fn new(gso: &'a mut MatGso, delta: f64, eta: f64, flags: i32) -> Self {
        use crate::defs::lll_flags::*;
        let siegel = (flags & SIEGEL) != 0;
        let early_red = (flags & EARLY_RED) != 0;
        let verbose = (flags & VERBOSE) != 0;
        let swap_threshold = if siegel { delta - eta * eta } else { delta };
        LllReduction {
            gso, delta, eta, swap_threshold, siegel, early_red, verbose,
            status: RedStatus::Success, final_kappa: 0, zeros: 0, n_swaps: 0,
        }
    }

    /// Babai size reduction: ensure |mu_true(kappa, j)| <= eta for j in [start, end).
    fn babai(&mut self, kappa: usize, end: usize, start: usize) -> bool {
        for _iter in 0..128 {
            if !self.gso.update_gso_row(kappa, end.saturating_sub(1)) {
                self.status = RedStatus::GsoFailure; return false;
            }

            let mut any_large = false;
            for j in start..end {
                if j >= kappa { continue; }
                let (m, e) = self.gso.get_mu_exp(kappa, j);
                let true_mu = crate::float::ldexp(m, e as i32);
                if true_mu.abs() > self.eta { any_large = true; break; }
            }
            if !any_large { return true; }

            // Snapshot stored mu values with their exponents
            let mut babai_mu = vec![0.0; end];
            let mut babai_expo = vec![0i64; end];
            for j in start..end {
                if j < kappa {
                    let (m, e) = self.gso.get_mu_exp(kappa, j);
                    babai_mu[j] = m;
                    babai_expo[j] = e;
                }
            }

            for j in (start..end).rev() {
                if j >= kappa { continue; }
                // Round mu stored at its native exponent. mu_m_ant = round(stored * 2^expo_bab).
                // If expo_bab > 0 the rounded integer may overflow i64; we split: rnd_we rounds
                // stored to integer at native precision, and the 2^expo_bab factor is emitted as expo.
                let (q_mantissa, q_expo) = round_we(babai_mu[j], babai_expo[j]);
                if q_mantissa == 0 { continue; }
                let q_true = (q_mantissa as f64) * 2f64.powi(q_expo as i32);
                // Update remaining babai_mu to reflect b[kappa] -= q_true * b[j].
                for k in start..j {
                    let (mjk, ejk) = self.gso.get_mu_exp(j, k);
                    let true_mjk = crate::float::ldexp(mjk, ejk as i32);
                    let old_true = crate::float::ldexp(babai_mu[k], babai_expo[k] as i32);
                    let new_true = old_true - q_true * true_mjk;
                    babai_mu[k] = new_true; babai_expo[k] = 0;
                }
                self.gso.row_addmul_2si(kappa, j, q_mantissa, q_expo);
            }
        }
        self.status = RedStatus::BabaiFailure; false
    }

    /// Driver: LLL reduce rows [kappa_min, kappa_end).
    pub fn lll(&mut self, kappa_min: usize, kappa_start: usize, kappa_end_opt: Option<usize>, size_reduction_start: usize) -> bool {
        let kappa_end = kappa_end_opt.unwrap_or(self.gso.d);
        let d_range = kappa_end.saturating_sub(kappa_min);
        if d_range == 0 { self.status = RedStatus::Success; return true; }

        // Push zero rows to the end.
        self.zeros = 0;
        while self.zeros < d_range && self.gso.b_row_is_zero(kappa_min) {
            self.gso.move_row(kappa_min, kappa_end - 1 - self.zeros);
            self.zeros += 1;
        }
        if self.zeros >= d_range { self.status = RedStatus::Success; return true; }

        // Ensure GSO is valid for row kappa_start.
        if !self.gso.update_gso_row(kappa_start, kappa_start) {
            self.status = RedStatus::GsoFailure; self.final_kappa = kappa_start; return false;
        }

        let mut kappa = (kappa_start + 1).max(kappa_min + 1);
        self.n_swaps = 0;

        // Iteration cap, generously set like fplll.
        let bmax = self.gso.get_max_exp_of_b().max(1) as f64;
        let d_eff = (d_range - self.zeros) as f64;
        let denom = (-self.delta.ln()).max(1e-6);
        let max_iter = ((d_eff * (d_eff + 1.0) * (bmax + 3.0) / denom) as u64)
            .saturating_add(32 * d_eff as u64)
            .saturating_add(4096);
        let mut iter: u64 = 0;

        while kappa < kappa_end - self.zeros {
            if iter >= max_iter { self.status = RedStatus::LllFailure; self.final_kappa = kappa; return false; }
            iter += 1;

            // Size reduction on kappa
            if !self.babai(kappa, kappa, size_reduction_start) {
                self.final_kappa = kappa; return false;
            }
            if !self.gso.update_gso_row(kappa, kappa) {
                self.status = RedStatus::GsoFailure; self.final_kappa = kappa; return false;
            }

            // Lovász: compare delta * ||b*_{k-1}||^2 with ||b*_k||^2 + mu^2 * ||b*_{k-1}||^2.
            // Under ROW_EXPO, r[i][i] is in scale 2^(2*row_expo[i]); reconcile scales.
            let (r_prev, e_prev) = self.gso.get_r_exp(kappa - 1, kappa - 1);
            let (r_cur, e_cur) = self.gso.get_r_exp(kappa, kappa);
            let (mu_stored, mu_expo) = self.gso.get_mu_exp(kappa, kappa - 1);
            let true_mu = crate::float::ldexp(mu_stored, mu_expo as i32);
            let shift = (e_prev - e_cur) as i32;
            let r_prev_scaled = crate::float::ldexp(r_prev, shift);
            let lhs = self.swap_threshold * r_prev_scaled;
            let rhs = if self.siegel { r_cur } else { r_cur + true_mu * true_mu * r_prev_scaled };

            if lhs > rhs {
                // Lovász fails: swap kappa-1, kappa and step back.
                self.gso.swap_rows(kappa - 1, kappa);
                self.n_swaps += 1;
                if kappa > kappa_min + 1 { kappa -= 1; }
            } else {
                kappa += 1;
            }
        }

        self.status = RedStatus::Success;
        self.final_kappa = kappa;
        true
    }
}

/// Entry: LLL-reduce the basis in `b` in place.
pub fn lll_reduce(b: &mut ZMatrix, delta: f64, eta: f64, flags: i32) -> RedStatus {
    let taken = std::mem::take(b);
    let mut gso = MatGso::new(taken, crate::gso::gso_flags::ROW_EXPO);
    let mut lll = LllReduction::new(&mut gso, delta, eta, flags);
    let _ = lll.lll(0, 0, None, 0);
    let status = lll.status;
    *b = gso.b;
    status
}

pub fn lll_reduce_default(b: &mut ZMatrix) -> RedStatus {
    lll_reduce(b, LLL_DEF_DELTA, LLL_DEF_ETA, 0)
}

/// Check (delta, eta)-LLL reducedness of the basis in `gso`.
pub fn is_lll_reduced(gso: &mut MatGso, delta: f64, eta: f64) -> bool {
    if !gso.update_gso() { return false; }
    for i in 0..gso.d {
        for j in 0..i {
            let (m, e) = gso.get_mu_exp(i, j);
            let true_mu = crate::float::ldexp(m, e as i32);
            if true_mu.abs() > eta + 1e-9 { return false; }
        }
    }
    for i in 1..gso.d {
        let (m, em) = gso.get_mu_exp(i, i - 1);
        let (r_prev, e_prev) = gso.get_r_exp(i - 1, i - 1);
        let (r_cur, e_cur) = gso.get_r_exp(i, i);
        // Bring prev into scale of e_cur.
        let shift = (e_prev - e_cur) as i32;
        let r_prev_scaled = crate::float::ldexp(r_prev, shift);
        let true_mu = crate::float::ldexp(m, em as i32);
        if r_cur + true_mu * true_mu * r_prev_scaled < delta * r_prev_scaled - 1e-9 * r_prev_scaled.abs() { return false; }
    }
    true
}

/// Helper: run `is_lll_reduced` on the given basis (constructs a row-expo-enabled GSO).
pub fn is_lll_reduced_basis(b: &ZMatrix, delta: f64, eta: f64) -> bool {
    let b2 = b.clone();
    let mut gso = MatGso::new(b2, crate::gso::gso_flags::ROW_EXPO);
    is_lll_reduced(&mut gso, delta, eta)
}
