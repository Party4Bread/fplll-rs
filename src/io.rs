//! Text I/O helpers for lattices and vectors.

use crate::integer::Z;
use crate::matrix::{parse_matrix, ZMatrix};
use std::fs;
use std::io;
use std::path::Path;

pub fn read_matrix_from_file<P: AsRef<Path>>(path: P) -> io::Result<ZMatrix> {
    let s = fs::read_to_string(path)?;
    parse_matrix(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn read_vector_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<Z>> {
    let s = fs::read_to_string(path)?;
    parse_vector(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn parse_vector(s: &str) -> Result<Vec<Z>, String> {
    // Accept `[a b c]` or `a b c` or newline-separated integers.
    let trimmed = s.trim();
    let inner = trimmed.strip_prefix('[').and_then(|t| t.strip_suffix(']')).unwrap_or(trimmed);
    let mut out = Vec::new();
    for tok in inner.split_whitespace() {
        let z = Z::from_str(tok).ok_or_else(|| format!("bad int '{}'", tok))?;
        out.push(z);
    }
    Ok(out)
}

pub fn write_vector(f: &mut impl std::io::Write, v: &[Z]) -> io::Result<()> {
    write!(f, "[")?;
    for (i, e) in v.iter().enumerate() {
        if i > 0 { write!(f, " ")?; }
        write!(f, "{}", e)?;
    }
    writeln!(f, "]")
}
