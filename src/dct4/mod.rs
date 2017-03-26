use rustfft::{FFTnum, Length};

mod dct4_fft;

pub trait DCT4<T: FFTnum>: Length {
	fn process(&mut self, input: &[T], output: &mut [T]);
}

pub use self::dct4_fft::DCT4ViaFFT;
