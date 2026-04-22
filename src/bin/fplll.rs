//! fplll CLI: a subset of the original tool's behaviour.
//!
//! Usage:
//!   fplll [-a lll|bkz|hlll|svp|cvp] [-d delta] [-e eta] [-b blocksize] [FILE]

use fplll::bkz::{bkz_reduce, BkzParam};
use fplll::defs::{LLL_DEF_DELTA, LLL_DEF_ETA};
use fplll::hlll::hlll_reduce;
use fplll::io::read_matrix_from_file;
use fplll::lll::lll_reduce;
use fplll::svpcvp::{closest_vector, shortest_vector};
use fplll::defs::SvpMethod;
use std::env;
use std::io::{self, BufRead, Write};

fn usage() -> ! {
    eprintln!("Usage: fplll [-a lll|bkz|hlll|svp|cvp] [-d delta] [-e eta] [-b blocksize] [FILE]");
    std::process::exit(1);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut action = "lll".to_string();
    let mut delta = LLL_DEF_DELTA;
    let mut eta = LLL_DEF_ETA;
    let mut block_size: usize = 20;
    let mut target_file: Option<String> = None;
    let mut file: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-a" => { action = args[i + 1].clone(); i += 2; }
            "-d" => { delta = args[i + 1].parse().unwrap_or(delta); i += 2; }
            "-e" => { eta = args[i + 1].parse().unwrap_or(eta); i += 2; }
            "-b" => { block_size = args[i + 1].parse().unwrap_or(block_size); i += 2; }
            "-t" => { target_file = Some(args[i + 1].clone()); i += 2; }
            "-h" | "--help" => usage(),
            s if s.starts_with('-') => { eprintln!("unknown flag {}", s); usage(); }
            _ => { file = Some(args[i].clone()); i += 1; }
        }
    }

    let mut b = if let Some(path) = &file {
        read_matrix_from_file(path)?
    } else {
        // Read from stdin
        let stdin = io::stdin();
        let mut s = String::new();
        for line in stdin.lock().lines() {
            s.push_str(&line?);
            s.push('\n');
        }
        fplll::matrix::parse_matrix(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    match action.as_str() {
        "lll" => {
            let _ = lll_reduce(&mut b, delta, eta, 0);
            b.write_to(&mut out)?;
        }
        "bkz" => {
            let mut p = BkzParam::default();
            p.block_size = block_size;
            p.delta = delta;
            let _ = bkz_reduce(&mut b, &p);
            b.write_to(&mut out)?;
        }
        "hlll" => {
            let _ = hlll_reduce(&mut b, delta, eta);
            b.write_to(&mut out)?;
        }
        "svp" => {
            match shortest_vector(&mut b, SvpMethod::Fast, 0) {
                Ok(coord) => {
                    // Print v = sum coord_i * b_i
                    let n = b.ncols();
                    let mut v = vec![fplll::integer::Z::zero(); n];
                    for (i, c) in coord.iter().enumerate() {
                        for j in 0..n {
                            v[j].addmul(c, b.get(i, j));
                        }
                    }
                    fplll::io::write_vector(&mut out, &v)?;
                }
                Err(e) => { eprintln!("SVP failed: {}", e); std::process::exit(1); }
            }
        }
        "cvp" => {
            let tf = target_file.unwrap_or_else(|| { eprintln!("cvp needs -t <target-file>"); std::process::exit(1); });
            let target = fplll::io::read_vector_from_file(&tf)?;
            let v = closest_vector(&mut b, &target).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            fplll::io::write_vector(&mut out, &v)?;
        }
        other => { eprintln!("unknown action {}", other); usage(); }
    }
    out.flush()?;
    Ok(())
}
