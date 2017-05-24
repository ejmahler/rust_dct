//! RustDCT is a pure-Rust signal processing library that computes the most common Discrete Cosine Transforms: 

//! * DCT Type 1
//! * DCT Type 2 (Often called "the" DCT - by far the most common algorithm, used by JPEG image compression and others)
//! * DCT Type 3 (the inverse of the DCT type 2, also used in JPEG)
//! * DCT Type 4
//! * MDCT (Used in audio and video compression such as Ogg and MP3)

//! The recommended way to use RustDCT is to create a `[DCTplanner](/blob/src/plan.rs)` instance, then call its `plan_dct1` or `plan_dct2` or etc method. Each DCT type has its own method which will choose the best algorithm for the given size.
//!
//! The recommended way to use RustFFT is to create a [`FFTplanner`](struct.FFTplanner.html) instance and then call its
//! `plan_fft` method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
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

pub mod dct1;
pub mod dct2;
pub mod dct3;
pub mod dct4;
pub mod mdct;
mod plan;
mod twiddles;

#[cfg(test)]
mod test_utils;

pub trait DCTnum: rustfft::FFTnum + Debug {}
impl DCTnum for f32 {}
impl DCTnum for f64 {}
pub use self::plan::DCTplanner;
