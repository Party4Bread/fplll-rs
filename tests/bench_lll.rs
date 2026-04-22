//! Benchmark: 20-dim integer-relations lattice with 1024-bit first column.
//!
//! fplll defaults (δ=0.99, η=0.51). Timed across 5 random seeds. Reports
//! f64+ROW_EXPO path and the MPFR path side by side.

use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::lll::lll_reduce;
use fplll::lll_mpfr::lll_reduce_mpfr;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;
use std::time::Instant;

fn gen(seed: u64, d: usize, bits: u32) -> ZMatrix {
    let mut rng = RandCtx::new_seeded(seed);
    let mut m = ZMatrix::new();
    m.resize(d, d + 1);
    m.gen_intrel(bits, &mut rng);
    m
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 { (v[n/2 - 1] + v[n/2]) / 2.0 } else { v[n/2] }
}

#[test]
fn bench_lll_20dim_1024bit() {
    let d = 20;
    let bits = 1024;
    let runs = 5;

    // f64 + ROW_EXPO path (may fail: 20-dim * 1024-bit → dot products ~2^2050 but normalised
    // per row to ~2^53, so the f64 path should work here).
    let mut f64_times = Vec::new();
    let mut f64_ok = 0;
    for seed in 0..runs {
        let mut m = gen(seed as u64, d, bits);
        let t = Instant::now();
        let st = lll_reduce(&mut m, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
        let dur = t.elapsed().as_secs_f64();
        if st as i32 == 0 { f64_ok += 1; f64_times.push(dur); }
        eprintln!("[f64]  seed {} -> {:?} in {:.3}s", seed, st, dur);
    }

    // MPFR path at 128-bit precision.
    let mut mpfr_times = Vec::new();
    let mut mpfr_ok = 0;
    for seed in 0..runs {
        let mut m = gen(seed as u64, d, bits);
        let t = Instant::now();
        let st = lll_reduce_mpfr(&mut m, 128, LLL_DEF_DELTA, LLL_DEF_ETA, 0);
        let dur = t.elapsed().as_secs_f64();
        if st as i32 == 0 { mpfr_ok += 1; mpfr_times.push(dur); }
        eprintln!("[mpfr] seed {} -> {:?} in {:.3}s", seed, st, dur);
    }

    eprintln!();
    eprintln!("=== LLL benchmark: d={} ({}x{}), bits={} ===", d, d, d + 1, bits);
    if !f64_times.is_empty() {
        let total: f64 = f64_times.iter().sum();
        eprintln!("  f64  : {}/{} succeeded, median {:.3}s, mean {:.3}s, min {:.3}s, max {:.3}s",
                  f64_ok, runs,
                  median(f64_times.clone()),
                  total / f64_times.len() as f64,
                  f64_times.iter().cloned().fold(f64::INFINITY, f64::min),
                  f64_times.iter().cloned().fold(0.0_f64, f64::max));
    } else {
        eprintln!("  f64  : all runs failed");
    }
    if !mpfr_times.is_empty() {
        let total: f64 = mpfr_times.iter().sum();
        eprintln!("  mpfr : {}/{} succeeded, median {:.3}s, mean {:.3}s, min {:.3}s, max {:.3}s",
                  mpfr_ok, runs,
                  median(mpfr_times.clone()),
                  total / mpfr_times.len() as f64,
                  mpfr_times.iter().cloned().fold(f64::INFINITY, f64::min),
                  mpfr_times.iter().cloned().fold(0.0_f64, f64::max));
    } else {
        eprintln!("  mpfr : all runs failed");
    }
}
