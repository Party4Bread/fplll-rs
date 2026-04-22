//! fplll-rs: Rust port of the fplll lattice reduction library.
//!
//! Provides LLL, HLLL, BKZ reduction, SVP/CVP solvers, Gram-Schmidt
//! orthogonalisation, lattice generators, and enumeration.
//!
//! The canonical C++ library: https://github.com/fplll/fplll
//!
//! This port uses [`rug::Integer`] for multi-precision integers and `f64`
//! for floating point computations by default. Most of fplll's algorithms
//! are implemented on top of this.

pub mod defs;
pub mod integer;
pub mod float;
pub mod float_mpfr;
pub mod matrix;
pub mod io;
pub mod util;
pub mod gso;
pub mod gso_mpfr;
pub mod lll;
pub mod lll_mpfr;
pub mod enumerate;
pub mod bkz;
pub mod pruner;
pub mod hlll;
pub mod svpcvp;
pub mod wrapper;
pub mod rand_util;

pub use defs::*;
pub use integer::{Integer, Z};
pub use float::FloatScalar;
pub use matrix::ZMatrix;
pub use gso::MatGso;
pub use lll::{lll_reduce, LllReduction, is_lll_reduced};
pub use bkz::{bkz_reduce, BkzParam, BkzAutoAbort};
pub use lll_mpfr::{lll_reduce_mpfr, lll_reduce_mpfr_default, is_lll_reduced_mpfr, LllReductionMpfr};
pub use gso_mpfr::MatGsoMpfr;
pub use hlll::{hlll_reduce, HLllReduction};
pub use svpcvp::{shortest_vector, closest_vector, shortest_vector_pruning};
pub use enumerate::{Enumeration, FastEvaluator};
pub use wrapper::Wrapper;
