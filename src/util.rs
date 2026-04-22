//! Miscellaneous utilities (volume, GH bound, precision estimates).

use crate::defs::LLL_DEF_EPSILON;
use crate::matrix::ZMatrix;
use crate::integer::Z;
use std::time::Instant;

thread_local! {
    static START: Instant = Instant::now();
}

pub fn cputime_ms() -> u64 {
    START.with(|s| s.elapsed().as_millis() as u64)
}

/// Minimum precision required so that the GSO error bounds remain valid.
/// Rough port of fplll::gso_min_prec / l2_min_prec.
pub fn gso_min_prec(d: i32, delta: f64, eta: f64, epsilon: f64) -> (u32, f64) {
    let eps = epsilon.min(eta - 0.5).min(1.0 - delta).max(1e-15);
    let rho = ((1.0 + eta).powi(2) + eps) / (delta - eta * eta).max(1e-15);
    let minprec = 5.0 + 2.0 * (d as f64).log2() - eps.log2() + (d as f64) * rho.log2();
    let prec = minprec.ceil().max(53.0) as u32;
    (prec, rho)
}

pub fn l2_min_prec(d: i32, delta: f64, eta: f64, epsilon: f64) -> u32 {
    let (p, _) = gso_min_prec(d, delta, eta, epsilon);
    p + 5 // L2 needs a little more headroom
}

/// Volume of the unit d-ball: Vol_d = pi^(d/2) / Gamma(d/2 + 1).
pub fn ball_volume(d: i32) -> f64 {
    use std::f64::consts::PI;
    let half = d as f64 / 2.0;
    PI.powf(half) / gamma(half + 1.0)
}

/// Gamma function (Lanczos approximation, good enough for GH estimates).
fn gamma(x: f64) -> f64 {
    // Reflection
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }
    let g = 7.0;
    let coef: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut a = coef[0];
    for (i, c) in coef.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
}

/// Gaussian-heuristic bound on the shortest vector (squared length),
/// given the b*-squared-norms r = (||b*_0||^2, ..., ||b*_{d-1}||^2).
pub fn gaussian_heuristic(r: &[f64], block_size: i32, root_det_log2: f64) -> f64 {
    // root_det = (prod r_i)^(1/bs); here caller supplies log2(root_det) already or 0 to derive.
    let b = block_size;
    let v = ball_volume(b);
    // GH^b = root_det^b / V_b  =>  GH = root_det / V_b^(1/b)
    let root_det = 2f64.powf(root_det_log2);
    let gh = root_det / v.powf(1.0 / b as f64);
    let _ = r;
    gh * gh
}

/// Squared Euclidean norm of a row of a ZMatrix (as f64, may lose precision for huge inputs).
pub fn row_sqnorm_f64(b: &ZMatrix, i: usize) -> f64 {
    let mut sq = Z::zero();
    for j in 0..b.ncols() { sq.addmul(b.get(i, j), b.get(i, j)); }
    let (m, e) = sq.get_d_2exp();
    m * 2f64.powi(e as i32)
}

pub fn default_epsilon() -> f64 { LLL_DEF_EPSILON }

pub fn get_red_status_str(code: i32) -> &'static str {
    match code {
        0 => "success",
        2 => "infinite number in GSO",
        3 => "infinite loop in babai",
        4 => "infinite loop in LLL",
        5 => "error in SVP solver",
        6 => "error in BKZ",
        7 => "time limit exceeded in BKZ",
        8 => "loops limit exceeded in BKZ",
        9 => "error in HLLL",
        10 => "increase of the norm",
        11 => "error in weak size reduction",
        _ => "(unknown)",
    }
}

/// Extend a vector to at least `new_len` elements, using `Default`.
pub fn extend_vec<T: Default + Clone>(v: &mut Vec<T>, new_len: usize) {
    if v.len() < new_len { v.resize(new_len, T::default()); }
}
