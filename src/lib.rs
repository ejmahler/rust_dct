//! RustDCT is a pure-Rust signal processing library that computes the most common Discrete Cosine Transforms
//!
//! * DCT Type 1
//! * DCT Type 2 (Often called "the" DCT - by far the most common algorithm, used by JPEG image compression and others)
//! * DCT Type 3 (the inverse of the DCT type 2, also used in JPEG)
//! * DCT Type 4
//! * MDCT (Used in audio and video compression such as Ogg and MP3)
//!
//! The recommended way to use RustDCT is to create a [`DCTplanner`](struct.DCTplanner.html) instance, then call its
//! `plan_dct1` or `plan_dct2` or etc methods. Each DCT type has its own method which will choose the best algorithm
//! for the given size.
//!
//! ```rust
//! // Compute a DCT Type 2 of size 1234
//! use rustdct::DCTplanner;
//!
//! let mut input:  Vec<f32> = vec![0f32; 1234];
//! let mut output: Vec<f32> = vec![0f32; 1234];
//!
//! let mut planner = DCTplanner::new();
//! let mut dct = planner.plan_dct2(1234);
//! dct.process(&mut input, &mut output);
//!
//! ```


pub extern crate rustfft;

#[cfg(test)]
extern crate rand;

use std::fmt::Debug;

pub use rustfft::num_complex;
pub use rustfft::num_traits;

/// Algorithms for computing the Discrete Cosine Transform Type 1
pub mod dct1;

/// Algorithms for computing the Discrete Cosine Transform Type 2
pub mod dct2;

/// Algorithms for computing the Discrete Cosine Transform Type 3
pub mod dct3;

/// Algorithms for computing the Discrete Cosine Transform Type 4
pub mod dct4;

/// Algorithms for computing the Modified Discrete Cosine Transform
pub mod mdct;
mod plan;
mod twiddles;

#[cfg(test)]
mod test_utils;

/// Generic floating point number, implemented for f32 and f64
pub trait DCTnum: rustfft::FFTnum + Debug {}
impl DCTnum for f32 {}
impl DCTnum for f64 {}

pub use self::plan::DCTplanner;
