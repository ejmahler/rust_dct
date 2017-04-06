use rustfft::Length;

use DCTnum;

mod dct2_via_fft;
mod dct2_naive;

pub trait DCT2<T: DCTnum>: Length {
	fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct2_via_fft::DCT2ViaFFT;
pub use self::dct2_naive::DCT2Naive;
