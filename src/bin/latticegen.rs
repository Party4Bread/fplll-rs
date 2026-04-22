//! latticegen: generate various families of lattice bases, compatible with
//! fplll's `latticegen` utility.

use fplll::integer::Z;
use fplll::matrix::ZMatrix;
use fplll::rand_util::RandCtx;
use std::env;
use std::io::{self, Write};

fn print_help() {
    eprintln!("Usage: latticegen [-randseed <int>|time] METHOD ARGS");
    eprintln!("Methods:");
    eprintln!("  r <d> <b>            integer relations, d x (d+1)");
    eprintln!("  s <d> <b> <b2>       simultaneous Diophantine");
    eprintln!("  u <d> <b>            uniform d x d");
    eprintln!("  n <d> <b|q> <c>      NTRU-like 2d x 2d");
    eprintln!("  N <d> <b|q> <c>      NTRU-like 2 2d x 2d");
    eprintln!("  q <d> <k> <b|q> <c>  q-ary");
    eprintln!("  t <d> <alpha>        triangular");
    std::process::exit(0);
}

fn die(msg: &str) -> ! { eprintln!("latticegen: {}", msg); std::process::exit(1); }

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" { print_help(); }

    let mut idx = 1;
    let mut rng = RandCtx::new_seeded(0);
    if args[idx] == "-randseed" {
        idx += 1;
        if args[idx] == "time" {
            rng = RandCtx::new_with_time();
        } else {
            let seed: u64 = args[idx].parse().unwrap_or_else(|_| die("bad seed"));
            rng = RandCtx::new_seeded(seed);
        }
        idx += 1;
    }

    if args.len() < idx + 2 { die("need method and dimension"); }
    let method = &args[idx]; idx += 1;
    let d: usize = args[idx].parse().unwrap_or_else(|_| die("bad dimension"));
    idx += 1;

    let mut m = ZMatrix::new();
    match method.as_str() {
        "r" => {
            if args.len() <= idx { die("method r needs <b>"); }
            let bits: u32 = args[idx].parse().unwrap_or_else(|_| die("bad b"));
            m.resize(d, d + 1);
            m.gen_intrel(bits, &mut rng);
        }
        "s" => {
            if args.len() <= idx + 1 { die("method s needs <b> <b2>"); }
            let b1: u32 = args[idx].parse().unwrap_or_else(|_| die("bad b"));
            let b2: u32 = args[idx + 1].parse().unwrap_or_else(|_| die("bad b2"));
            m.resize(d + 1, d + 1);
            m.gen_simdioph(b1, b2, &mut rng);
        }
        "u" => {
            if args.len() <= idx { die("method u needs <b>"); }
            let bits: u32 = args[idx].parse().unwrap_or_else(|_| die("bad b"));
            m.resize(d, d);
            m.gen_uniform(bits, &mut rng);
        }
        "n" | "N" => {
            if args.len() <= idx + 1 { die("method n/N needs <b|q> <c>"); }
            let flag = args[idx + 1].as_str();
            m.resize(2 * d, 2 * d);
            let q = match flag {
                "b" => {
                    let bits: u32 = args[idx].parse().unwrap_or_else(|_| die("bad b"));
                    ZMatrix::gen_q(bits, &mut rng)
                }
                "q" => {
                    Z::from_str(&args[idx]).unwrap_or_else(|| die("bad q"))
                }
                _ => die("c must be 'b' or 'q'"),
            };
            if method == "n" { m.gen_ntrulike(&q, &mut rng); } else { m.gen_ntrulike2(&q, &mut rng); }
        }
        "q" => {
            if args.len() <= idx + 2 { die("method q needs <k> <b|q> <c>"); }
            let k: usize = args[idx].parse().unwrap_or_else(|_| die("bad k"));
            let flag = args[idx + 2].as_str();
            m.resize(d, d);
            let q = match flag {
                "b" => {
                    let bits: u32 = args[idx + 1].parse().unwrap_or_else(|_| die("bad b"));
                    ZMatrix::gen_q(bits, &mut rng)
                }
                "q" => Z::from_str(&args[idx + 1]).unwrap_or_else(|| die("bad q")),
                "p" => {
                    let bits: u32 = args[idx + 1].parse().unwrap_or_else(|_| die("bad b"));
                    ZMatrix::gen_q(bits, &mut rng).next_prime()
                }
                _ => die("c must be 'b'/'q'/'p'"),
            };
            m.gen_qary(k, &q, &mut rng);
        }
        "t" => {
            if args.len() <= idx { die("method t needs <alpha>"); }
            let alpha: f64 = args[idx].parse().unwrap_or_else(|_| die("bad alpha"));
            m.resize(d, d);
            m.gen_trg(alpha, &mut rng);
        }
        _ => die("invalid method"),
    }

    let stdout = io::stdout();
    let mut w = stdout.lock();
    m.write_to(&mut w)?;
    w.flush()?;
    Ok(())
}
