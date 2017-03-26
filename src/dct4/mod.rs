use rustfft::Length;

use DCTnum;

mod dct4_fft;
mod dct4_naive;

pub trait DCT4<T: DCTnum>: Length {
	fn process(&mut self, input: &[T], output: &mut [T]);
}

pub use self::dct4_fft::DCT4ViaFFT;
pub use self::dct4_naive::DCT4Naive;
