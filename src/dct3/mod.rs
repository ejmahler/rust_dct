use rustfft::Length;

use DCTnum;

mod dct3_via_fft;
mod dct3_naive;

pub trait DCT3<T: DCTnum>: Length {
	fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct3_via_fft::DCT3ViaFFT;
pub use self::dct3_naive::DCT3Naive;
