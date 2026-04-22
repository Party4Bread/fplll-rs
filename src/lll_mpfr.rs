//! LLL reduction using MPFR (rug::Float) precision for the Gram-Schmidt
//! coefficients. Slower than the f64 path but correct for bases where
//! f64 precision is insufficient.

use crate::defs::{LLL_DEF_DELTA, LLL_DEF_ETA, RedStatus, SIZE_RED_FAILURE_THRESH};
use crate::float_mpfr as fp;
use crate::gso_mpfr::MatGsoMpfr;
use crate::matrix::ZMatrix;
use rug::{Assign, Float};

pub struct LllReductionMpfr<'a> {
    gso: &'a mut MatGsoMpfr,
    pub delta: Float,
    pub eta: Float,
    pub siegel: bool,
    pub status: RedStatus,
    pub n_swaps: u64,
}

impl<'a> LllReductionMpfr<'a> {
    pub fn new(gso: &'a mut MatGsoMpfr, delta: f64, eta: f64, flags: i32) -> Self {
        use crate::defs::lll_flags::*;
        let siegel = (flags & SIEGEL) != 0;
        let prec = gso.prec;
        let delta_f = Float::with_val(prec, delta);
        let eta_f = Float::with_val(prec, eta);
        LllReductionMpfr {
            gso, delta: delta_f, eta: eta_f, siegel,
            status: RedStatus::Success,
            n_swaps: 0,
        }
    }

    fn babai(&mut self, kappa: usize, end: usize, start: usize) -> bool {
        let prec = self.gso.prec;
        let _old = fp::set_prec(prec);
        for _iter in 0..128 {
            if !self.gso.update_gso_row(kappa, end.saturating_sub(1)) {
                self.status = RedStatus::GsoFailure; return false;
            }
            let mut any_large = false;
            for j in start..end {
                if j >= kappa { continue; }
                let m = self.gso.get_mu(kappa, j);
                if fp::abs(m) > self.eta { any_large = true; break; }
            }
            if !any_large { return true; }

            let mut babai_mu: Vec<Float> = (0..end).map(|_| fp::zero()).collect();
            for j in start..end {
                if j < kappa { babai_mu[j].assign(self.gso.get_mu(kappa, j)); }
            }

            for j in (start..end).rev() {
                if j >= kappa { continue; }
                let q = fp::to_integer_rnd(&babai_mu[j]);
                if q == 0 { continue; }
                let qf = Float::with_val(prec, &q);
                for k in start..j {
                    let prod = Float::with_val(prec, &qf * self.gso.get_mu(j, k));
                    babai_mu[k] -= prod;
                }
                self.gso.row_addmul_z(kappa, j, &q);
            }
        }
        self.status = RedStatus::BabaiFailure; false
    }

    pub fn lll(&mut self, kappa_min: usize, kappa_start: usize, kappa_end_opt: Option<usize>, size_reduction_start: usize) -> bool {
        let kappa_end = kappa_end_opt.unwrap_or(self.gso.d);
        let d_range = kappa_end.saturating_sub(kappa_min);
        if d_range == 0 { self.status = RedStatus::Success; return true; }

        let mut zeros = 0;
        while zeros < d_range && self.gso.b_row_is_zero(kappa_min) {
            self.gso.move_row(kappa_min, kappa_end - 1 - zeros);
            zeros += 1;
        }
        if zeros >= d_range { self.status = RedStatus::Success; return true; }

        if !self.gso.update_gso_row(kappa_start, kappa_start) {
            self.status = RedStatus::GsoFailure; return false;
        }

        let mut kappa = (kappa_start + 1).max(kappa_min + 1);
        let prec = self.gso.prec;

        let bmax = self.gso.get_max_exp_of_b().max(1) as f64;
        let delta_d = self.delta.to_f64();
        let denom = (-delta_d.ln()).max(1e-6);
        let max_iter = (((d_range as f64) * (d_range as f64 + 1.0) * (bmax + 3.0) / denom) as u64)
            .saturating_add(64 * d_range as u64).saturating_add(8192);
        let mut iter: u64 = 0;

        while kappa < kappa_end - zeros {
            if iter >= max_iter { self.status = RedStatus::LllFailure; return false; }
            iter += 1;
            if !self.babai(kappa, kappa, size_reduction_start) { return false; }
            if !self.gso.update_gso_row(kappa, kappa) {
                self.status = RedStatus::GsoFailure; return false;
            }
            let r_prev = self.gso.r_diag(kappa - 1).clone();
            let r_cur = self.gso.r_diag(kappa).clone();
            let mu = self.gso.get_mu(kappa, kappa - 1).clone();

            let lhs = Float::with_val(prec, &self.delta * &r_prev);
            let rhs = if self.siegel {
                r_cur.clone()
            } else {
                let mu2 = Float::with_val(prec, &mu * &mu);
                let mu2_rp = Float::with_val(prec, &mu2 * &r_prev);
                Float::with_val(prec, &r_cur + &mu2_rp)
            };
            if lhs > rhs {
                self.gso.swap_rows(kappa - 1, kappa);
                self.n_swaps += 1;
                if kappa > kappa_min + 1 { kappa -= 1; }
            } else {
                kappa += 1;
            }
        }
        self.status = RedStatus::Success;
        let _ = SIZE_RED_FAILURE_THRESH; // silence unused import
        true
    }
}

pub fn lll_reduce_mpfr(b: &mut ZMatrix, prec: u32, delta: f64, eta: f64, flags: i32) -> RedStatus {
    let taken = std::mem::take(b);
    let mut gso = MatGsoMpfr::new(taken, prec);
    let mut lll = LllReductionMpfr::new(&mut gso, delta, eta, flags);
    let _ = lll.lll(0, 0, None, 0);
    let status = lll.status;
    *b = gso.b;
    status
}

pub fn lll_reduce_mpfr_default(b: &mut ZMatrix, prec: u32) -> RedStatus {
    lll_reduce_mpfr(b, prec, LLL_DEF_DELTA, LLL_DEF_ETA, 0)
}

/// Check (delta, eta)-LLL reducedness using MPFR precision.
pub fn is_lll_reduced_mpfr(b: &ZMatrix, prec: u32, delta: f64, eta: f64) -> bool {
    let mut gso = MatGsoMpfr::new(b.clone(), prec);
    if !gso.update_gso() { return false; }
    let eta_f = Float::with_val(prec, eta);
    let delta_f = Float::with_val(prec, delta);
    for i in 0..gso.d {
        for j in 0..i {
            if fp::abs(gso.get_mu(i, j)) > eta_f {
                let slack = Float::with_val(prec, 1e-9);
                if fp::abs(gso.get_mu(i, j)) > Float::with_val(prec, &eta_f + &slack) { return false; }
            }
        }
    }
    for i in 1..gso.d {
        let mu = gso.get_mu(i, i - 1).clone();
        let r_prev = gso.r_diag(i - 1).clone();
        let r_cur = gso.r_diag(i).clone();
        let mu2 = Float::with_val(prec, &mu * &mu);
        let sum = Float::with_val(prec, &r_cur + &Float::with_val(prec, &mu2 * &r_prev));
        let target = Float::with_val(prec, &delta_f * &r_prev);
        if sum < target {
            let tol = Float::with_val(prec, 1e-9) * r_prev.clone();
            if sum < Float::with_val(prec, &target - &tol) { return false; }
        }
    }
    true
}
