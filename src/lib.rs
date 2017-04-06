
extern crate num;
extern crate rustfft;

#[cfg(test)]
extern crate rand;

use std::fmt::Debug;

mod dct_type_2;
pub mod dct3;
pub mod dct4;
mod mdct;
mod plan;
mod twiddles;

#[cfg(test)]
mod test_utils;

pub trait DCTnum: rustfft::FFTnum + Debug {}
impl DCTnum for f32 {}
impl DCTnum for f64 {}

pub use self::dct_type_2::DCT2;
pub use self::mdct::MDCT;
pub use self::mdct::window_fn;
pub use self::plan::DCTPlanner;
