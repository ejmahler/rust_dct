use rustfft::Length;

use DCTnum;

mod dct1_naive;

pub trait DCT1<T: DCTnum>: Length {
	fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct1_naive::DCT1Naive;
