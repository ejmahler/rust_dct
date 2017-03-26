
extern crate num;
extern crate rustfft;

#[cfg(test)]
extern crate rand;

mod dct_type_2;
mod dct_type_3;
pub mod dct4;
mod mdct;
mod plan;

#[cfg(test)]
mod test_utils;

pub trait DCTnum: rustfft::FFTnum {}
impl DCTnum for f32 {}
impl DCTnum for f64 {}

pub use self::dct_type_2::DCT2;
pub use self::dct_type_3::DCT3;
pub use self::mdct::MDCT;
pub use self::mdct::window_fn;
pub use self::plan::DCTPlanner;
