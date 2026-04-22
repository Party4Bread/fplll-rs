//! Constants and enumerations mirroring fplll/defs.h.

pub const LLL_DEF_DELTA: f64 = 0.99;
pub const LLL_DEF_ETA: f64 = 0.51;
pub const LLL_DEF_EPSILON: f64 = 0.01;
pub const SIZE_RED_FAILURE_THRESH: i64 = 5;

pub const HLLL_DEF_THETA: f64 = 0.001;
pub const HLLL_DEF_C: f64 = 0.1;

pub const MAX_EXP_DOUBLE: i64 = 1000;
pub const PREC_DOUBLE: u32 = 53;
pub const CPU_SIZE_1: i64 = 53;
pub const MAX_LONG_FAST: f64 = (1u64 << 53) as f64;

pub const BKZ_DEF_AUTO_ABORT_SCALE: f64 = 1.0;
pub const BKZ_DEF_AUTO_ABORT_MAX_NO_DEC: i32 = 5;
pub const BKZ_DEF_GH_FACTOR: f64 = 1.1;
pub const BKZ_DEF_MIN_SUCCESS_PROBABILITY: f64 = 0.5;
pub const BKZ_DEF_RERANDOMIZATION_DENSITY: i32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RedStatus {
    Success = 0,
    GsoFailure = 2,
    BabaiFailure = 3,
    LllFailure = 4,
    EnumFailure = 5,
    BkzFailure = 6,
    BkzTimeLimit = 7,
    BkzLoopsLimit = 8,
    HlllFailure = 9,
    HlllNormFailure = 10,
    HlllSrFailure = 11,
}

impl RedStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RedStatus::Success => "success",
            RedStatus::GsoFailure => "infinite number in GSO",
            RedStatus::BabaiFailure => "infinite loop in babai",
            RedStatus::LllFailure => "infinite loop in LLL",
            RedStatus::EnumFailure => "error in SVP solver",
            RedStatus::BkzFailure => "error in BKZ",
            RedStatus::BkzTimeLimit => "time limit exceeded in BKZ",
            RedStatus::BkzLoopsLimit => "loops limit exceeded in BKZ",
            RedStatus::HlllFailure => "error in HLLL",
            RedStatus::HlllNormFailure => "increase of the norm",
            RedStatus::HlllSrFailure => "error in weak size reduction",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LllMethod {
    Wrapper = 0,
    Proved = 1,
    Heuristic = 2,
    Fast = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    Default = 0,
    Double = 1,
    LongDouble = 2,
    Dpe = 3,
    Dd = 4,
    Qd = 5,
    Mpfr = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntType {
    Mpz = 0,
    Long = 1,
    Double = 2,
}

pub mod lll_flags {
    pub const DEFAULT: i32 = 0;
    pub const VERBOSE: i32 = 1;
    pub const EARLY_RED: i32 = 2;
    pub const SIEGEL: i32 = 4;
}

pub mod svp_flags {
    pub const DEFAULT: i32 = 0;
    pub const VERBOSE: i32 = 1;
    pub const OVERRIDE_BND: i32 = 2;
    pub const DUAL: i32 = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvpMethod {
    Fast = 0,
    Proved = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvpMethod {
    Fast = 0,
    Proved = 2,
}

pub mod bkz_flags {
    pub const DEFAULT: i32 = 0;
    pub const VERBOSE: i32 = 1;
    pub const NO_LLL: i32 = 2;
    pub const MAX_LOOPS: i32 = 4;
    pub const MAX_TIME: i32 = 8;
    pub const BOUNDED_LLL: i32 = 0x10;
    pub const AUTO_ABORT: i32 = 0x20;
    pub const DUMP_GSO: i32 = 0x40;
    pub const GH_BND: i32 = 0x80;
    pub const SD_VARIANT: i32 = 0x100;
    pub const SLD_RED: i32 = 0x200;
}
