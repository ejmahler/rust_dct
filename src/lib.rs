
pub extern crate rustfft;

#[cfg(test)]
extern crate rand;

use std::fmt::Debug;

pub use rustfft::{num_complex, num_traits};

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

pub use self::mdct::MDCT;
pub use self::mdct::window_fn;
pub use self::plan::DCTPlanner;
