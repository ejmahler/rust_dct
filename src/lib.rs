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
//! use std::sync::Arc;
//! use rustdct::DCTplanner;
//!
//! let mut input:  Vec<f32> = vec![0f32; 1234];
//! let mut output: Vec<f32> = vec![0f32; 1234];
//!
//! let mut planner = DCTplanner::new();
//! let dct = planner.plan_dct2(1234);
//! dct.process(&mut input, &mut output);
//!
//! // The DCT instance returned by the planner is stored behind an `Arc`, so it's cheap to clone
//! let dct_clone = Arc::clone(&dct);
//! ```
//! 
//! RustDCT also exposes individual DCT algorithms. For example, if you're writing a JPEG compression library, it's
//! safe to assume you want a DCT2 and DCT3 of size 8. Instead of going through the planner, you can directly create
//! hardcoded DCT instances of size 8.
//! ```rust
//! // Compute a DCT type 2 of size 8, and then compute a DCT type 3 of size 8 on the output.
//! use rustdct::dct2::DCT2;
//! use rustdct::dct2::dct2_butterflies::DCT2Butterfly8;
//! use rustdct::dct3::DCT3;
//! use rustdct::dct3::dct3_butterflies::DCT3Butterfly8;
//! 
//! let mut input = [0f32; 8];
//! let mut intermediate = [0f32; 8];
//! let mut output = [0f32; 8];
//! 
//! let dct2 = DCT2Butterfly8::new();
//! let dct3 = DCT3Butterfly8::new();
//! 
//! dct2.process(&mut input, &mut intermediate);
//! dct3.process(&mut intermediate, &mut output);
//! ```


pub extern crate rustfft;

#[cfg(test)]
extern crate rand;


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
mod common;
pub use common::DCTnum;

pub use self::plan::DCTplanner;

#[cfg(test)]
mod test_utils;
