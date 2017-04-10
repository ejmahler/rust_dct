use rustfft::Length;

use DCTnum;

mod mdct_naive;
mod mdct_via_dct4;

mod imdct_naive;
mod imdct_via_dct4;
pub mod window_fn;

pub trait MDCT<T: DCTnum>: Length {
	fn process_split(&mut self, input_a: &[T], input_b: &[T], output: &mut [T]);
	fn process(&mut self, input: &[T], output: &mut [T]) {
		let (input_a, input_b) = input.split_at(output.len());

		self.process_split(input_a, input_b, output);
	}
}

pub trait IMDCT<T: DCTnum>: Length {
	fn process_split(&mut self, input: &[T], output_a: &mut [T], output_b: &mut [T]);
	fn process(&mut self, input: &[T], output: &mut [T]) {
		let (output_a, output_b) = output.split_at_mut(input.len());

		self.process_split(input, output_a, output_b);
	}
}

pub use self::mdct_naive::MDCTNaive;
pub use self::mdct_via_dct4::MDCTViaDCT4;

pub use self::imdct_naive::IMDCTNaive;
pub use self::imdct_via_dct4::IMDCTViaDCT4;
