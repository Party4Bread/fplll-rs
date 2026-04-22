//! Row-major dense matrix type, specialised for integer lattice bases.
//!
//! Mirrors fplll's `ZZ_mat<mpz_t>` with the row-operation API used by GSO.

use crate::integer::Z;
use crate::rand_util::{rand_bits, rand_below, RandCtx};
use rug::Integer as RugInt;
use std::fmt;
use std::io::{self, BufRead};

/// A row of a `ZMatrix`. Owned `Vec<Z>`.
pub type ZRow = Vec<Z>;

/// Row-major integer matrix. The lattice basis `b` is a `ZMatrix`.
#[derive(Clone, Default)]
pub struct ZMatrix {
    rows: Vec<ZRow>,
    ncols: usize,
}

impl ZMatrix {
    pub fn new() -> Self { ZMatrix { rows: Vec::new(), ncols: 0 } }
    pub fn from_dims(r: usize, c: usize) -> Self {
        let mut m = ZMatrix { rows: Vec::with_capacity(r), ncols: c };
        for _ in 0..r { m.rows.push(vec![Z::zero(); c]); }
        m
    }
    pub fn nrows(&self) -> usize { self.rows.len() }
    pub fn ncols(&self) -> usize { self.ncols }

    pub fn resize(&mut self, r: usize, c: usize) {
        self.ncols = c;
        if self.rows.len() > r {
            self.rows.truncate(r);
        }
        for row in &mut self.rows {
            row.resize(c, Z::zero());
        }
        while self.rows.len() < r {
            self.rows.push(vec![Z::zero(); c]);
        }
    }

    pub fn row(&self, i: usize) -> &ZRow { &self.rows[i] }
    pub fn row_mut(&mut self, i: usize) -> &mut ZRow { &mut self.rows[i] }
    pub fn get(&self, i: usize, j: usize) -> &Z { &self.rows[i][j] }
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Z { &mut self.rows[i][j] }
    pub fn set(&mut self, i: usize, j: usize, v: Z) { self.rows[i][j] = v; }

    pub fn fill(&mut self, value: i64) {
        for r in &mut self.rows {
            for e in r.iter_mut() { e.set_si(value); }
        }
    }

    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b { self.rows.swap(a, b); }
    }

    /// (m[first], ..., m[last]) -> (m[first+1], ..., m[last], m[first])
    pub fn rotate_left(&mut self, first: usize, last: usize) {
        for i in first..last { self.rows.swap(i, i + 1); }
    }
    /// (m[first], ..., m[last]) -> (m[last], m[first], ..., m[last-1])
    pub fn rotate_right(&mut self, first: usize, last: usize) {
        let mut i = last;
        while i > first { self.rows.swap(i, i - 1); i -= 1; }
    }

    pub fn gen_zero(&mut self, d: usize, n: usize) {
        self.resize(d, n);
        self.fill(0);
    }

    pub fn gen_identity(&mut self, d: usize) {
        self.gen_zero(d, d);
        for i in 0..d { self.rows[i][i] = Z::one(); }
    }

    /// Integer relation lattice: d x (d+1) matrix with a random first column, identity in the rest.
    pub fn gen_intrel(&mut self, bits: u32, rng: &mut RandCtx) {
        assert!(self.nrows() > 0 && self.ncols() == self.nrows() + 1);
        let d = self.nrows();
        for i in 0..d {
            self.rows[i][0] = Z(rand_bits(rng, bits));
            for j in 1..d + 1 {
                self.rows[i][j].set_si(if i + 1 == j { 1 } else { 0 });
            }
        }
    }

    /// Simultaneous Diophantine: (d+1) x (d+1) with diagonal 1s and a random row.
    pub fn gen_simdioph(&mut self, bits: u32, bits2: u32, rng: &mut RandCtx) {
        let d = self.nrows() - 1;
        // clear
        for i in 0..=d { for j in 0..=d { self.rows[i][j].set_si(0); } }
        // first row: 2^bits2 at (0,0), random mod 2^bits on the rest
        self.rows[0][0] = Z(RugInt::from(1) << bits2);
        for j in 1..=d { self.rows[0][j] = Z(rand_bits(rng, bits)); }
        // identity * 2^bits below
        let two_b = RugInt::from(1) << bits;
        for i in 1..=d { self.rows[i][i] = Z(two_b.clone()); }
    }

    /// Uniform random d x d matrix with entries in [0, 2^bits).
    pub fn gen_uniform(&mut self, bits: u32, rng: &mut RandCtx) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self.rows[i][j] = Z(rand_bits(rng, bits));
            }
        }
    }

    /// Choose a random q in [2^(bits-1), 2^bits - 1].
    pub fn gen_q(bits: u32, rng: &mut RandCtx) -> Z {
        let pow2 = RugInt::from(1) << (bits - 1);
        let r = rand_bits(rng, bits - 1);
        Z(r + pow2)
    }

    /// Construct [[I, H], [0, qI]] of size 2d x 2d where H is a rotation matrix of a random vector.
    pub fn gen_ntrulike(&mut self, q: &Z, rng: &mut RandCtx) {
        let d = self.nrows() / 2;
        self.fill(0);
        // identity top-left
        for i in 0..d { self.rows[i][i] = Z::one(); }
        // h[] random mod q
        let mut h = vec![Z::zero(); d];
        for i in 0..d { h[i] = Z(rand_below(rng, &q.0)); }
        // top-right = rotation of h
        for i in 0..d {
            for j in 0..d {
                let k = (i + j) % d;
                self.rows[i][d + j] = h[k].clone();
            }
        }
        // bottom-right = q*I
        for i in 0..d { self.rows[d + i][d + i] = q.clone(); }
    }

    /// Construct [[qI, 0], [H, I]] of size 2d x 2d.
    pub fn gen_ntrulike2(&mut self, q: &Z, rng: &mut RandCtx) {
        let d = self.nrows() / 2;
        self.fill(0);
        for i in 0..d { self.rows[i][i] = q.clone(); }
        let mut h = vec![Z::zero(); d];
        for i in 0..d { h[i] = Z(rand_below(rng, &q.0)); }
        for i in 0..d {
            for j in 0..d {
                let k = (i + j) % d;
                self.rows[d + i][j] = h[k].clone();
            }
        }
        for i in 0..d { self.rows[d + i][d + i] = Z::one(); }
    }

    /// Construct [[I, H], [0, qI]] where H is uniform mod q, (n-k) x k.
    pub fn gen_qary(&mut self, k: usize, q: &Z, rng: &mut RandCtx) {
        let n = self.nrows();
        assert!(k <= n);
        self.fill(0);
        let top = n - k;
        for i in 0..top { self.rows[i][i] = Z::one(); }
        for i in 0..top {
            for j in 0..k {
                self.rows[i][top + j] = Z(rand_below(rng, &q.0));
            }
        }
        for i in 0..k { self.rows[top + i][top + i] = q.clone(); }
    }

    /// Lower triangular with diag 2^{floor(alpha*(2i-d))} and random sub-diagonal bounded by diag.
    pub fn gen_trg(&mut self, alpha: f64, rng: &mut RandCtx) {
        let d = self.nrows();
        assert!(self.ncols() == d);
        self.fill(0);
        for i in 0..d {
            let exp = (alpha * (2 * (i as i64) - d as i64) as f64).floor() as i64;
            let e = exp.max(0) as u32;
            let diag = RugInt::from(1) << e;
            self.rows[i][i] = Z(diag.clone());
            for j in 0..i {
                // uniform in (-diag/2, diag/2)
                let r = rand_below(rng, &diag);
                let half = diag.clone() >> 1u32;
                self.rows[i][j] = Z(r - half);
            }
        }
    }

    pub fn transpose(&mut self) {
        let r = self.nrows();
        let c = self.ncols();
        let mut nm = ZMatrix::from_dims(c, r);
        for i in 0..r {
            for j in 0..c {
                nm.rows[j][i] = self.rows[i][j].clone();
            }
        }
        *self = nm;
    }

    /// Read from text format "[[a b c] [d e f] ...]" (fplll matrix format).
    pub fn read_from<R: BufRead>(r: &mut R) -> io::Result<ZMatrix> {
        let mut buf = String::new();
        r.read_to_string(&mut buf)?;
        parse_matrix(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn write_to(&self, f: &mut impl std::io::Write) -> io::Result<()> {
        writeln!(f, "[")?;
        for row in &self.rows {
            write!(f, "[")?;
            let mut first = true;
            for v in row {
                if !first { write!(f, " ")?; }
                first = false;
                write!(f, "{}", v)?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f, "]")?;
        Ok(())
    }

    pub fn max_exp_of_b(&self) -> i64 {
        let mut m: i64 = 0;
        for r in &self.rows {
            for v in r {
                if v.is_zero() { continue; }
                let b = v.0.significant_bits() as i64;
                if b > m { m = b; }
            }
        }
        m
    }

    /// true if row `i` is all zero.
    pub fn b_row_is_zero(&self, i: usize) -> bool {
        self.rows[i].iter().all(|x| x.is_zero())
    }

    /// Dot product of row i and row j (full length).
    pub fn row_dot(&self, i: usize, j: usize) -> Z {
        let mut out = Z::zero();
        for k in 0..self.ncols() {
            out.addmul(&self.rows[i][k], &self.rows[j][k]);
        }
        out
    }

    /// Truncated dot product: sum over k in [0, n).
    pub fn row_dot_trunc(&self, i: usize, j: usize, n: usize) -> Z {
        let mut out = Z::zero();
        for k in 0..n { out.addmul(&self.rows[i][k], &self.rows[j][k]); }
        out
    }

    /// b[i] += x * b[j], length n
    pub fn row_addmul_vec(&mut self, i: usize, j: usize, x: &Z, n: usize) {
        if i == j { return; }
        // Split: borrow both rows via split_at_mut
        let (lo, hi) = if i < j { self.rows.split_at_mut(j) } else { self.rows.split_at_mut(i) };
        let (ri, rj) = if i < j { (&mut lo[i], &hi[0]) } else { (&mut hi[0], &lo[j]) };
        for k in 0..n { ri[k].addmul(x, &rj[k]); }
    }

    /// b[i] += (x * 2^e) * b[j], length n
    pub fn row_addmul_2si_vec(&mut self, i: usize, j: usize, x: &Z, e: i64, n: usize) {
        if i == j { return; }
        let mut scaled = Z::zero();
        scaled.mul_2si(x, e);
        self.row_addmul_vec(i, j, &scaled, n);
    }

    pub fn negate_row(&mut self, i: usize) {
        for e in &mut self.rows[i] { e.neg_inplace(); }
    }

    /// Hadamard ratio diagnostic (sum log ||b_i||).
    pub fn log_norm_sum(&self) -> f64 {
        let mut s = 0.0;
        for r in &self.rows {
            let mut sq = Z::zero();
            for v in r { sq.addmul(v, v); }
            let (m, e) = sq.get_d_2exp();
            if m > 0.0 { s += (m.log2() + e as f64) * 0.5; }
        }
        s
    }
}

impl fmt::Display for ZMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for row in &self.rows {
            write!(f, "[")?;
            let mut first = true;
            for v in row {
                if !first { write!(f, " ")?; }
                first = false;
                write!(f, "{}", v)?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for ZMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZMatrix({}x{})\n{}", self.nrows(), self.ncols(), self)
    }
}

/// Parse fplll-style matrix text: `[[a b ...] [c d ...]]` (whitespace/newlines flexible).
pub fn parse_matrix(s: &str) -> Result<ZMatrix, String> {
    // Tokenise into brackets and numbers.
    let mut tokens: Vec<&str> = Vec::new();
    let mut i = 0usize;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'[' || c == b']' {
            tokens.push(std::str::from_utf8(&bytes[i..i + 1]).unwrap());
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'[' && bytes[i] != b']' {
                i += 1;
            }
            tokens.push(std::str::from_utf8(&bytes[start..i]).unwrap());
        }
    }

    let mut p = 0usize;
    if p >= tokens.len() || tokens[p] != "[" { return Err("expected '['".into()); }
    p += 1;
    let mut rows: Vec<ZRow> = Vec::new();
    let mut ncols = 0usize;
    while p < tokens.len() && tokens[p] == "[" {
        p += 1;
        let mut row: ZRow = Vec::new();
        while p < tokens.len() && tokens[p] != "]" {
            let z = Z::from_str(tokens[p]).ok_or_else(|| format!("bad int '{}'", tokens[p]))?;
            row.push(z);
            p += 1;
        }
        if p >= tokens.len() { return Err("missing ']'".into()); }
        p += 1; // ']'
        if rows.is_empty() { ncols = row.len(); }
        else if row.len() != ncols { return Err(format!("ragged row: expected {}, got {}", ncols, row.len())); }
        rows.push(row);
    }
    if p >= tokens.len() || tokens[p] != "]" { return Err("expected closing ']'".into()); }
    Ok(ZMatrix { rows, ncols })
}
