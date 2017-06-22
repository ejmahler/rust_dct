use rustfft::Length;

use DCTnum;

mod mdct_naive;
mod mdct_via_dct4;

mod imdct_naive;
mod imdct_via_dct4;
pub mod window_fn;

/// An umbrella trait for algorithms which compute the Modified Discrete Cosine Transform (MDCT)
pub trait MDCT<T: DCTnum>: Length {
    /// Computes the MDCT on the `input` buffer and places the result in the `output` buffer.
    ///
    /// To make overlapping array segments easier, this method DOES NOT modify the input buffer.
    fn process(&mut self, input: &[T], output: &mut [T]) {
        let (input_a, input_b) = input.split_at(output.len());

        self.process_split(input_a, input_b, output);
    }

    /// Computes the MDCT on the `input` buffer and places the result in the `output` buffer.
    /// Uses `input_a` for the first half of the input, and `input_b` for the second half of the input
    ///
    /// To make overlapping array segments easier, this method DOES NOT modify the input buffer.
    fn process_split(&mut self, input_a: &[T], input_b: &[T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Inverse Modified Discrete Cosine Transform (IMDCT)
pub trait IMDCT<T: DCTnum>: Length {
    /// Computes the MDCT on the `input` buffer and places the result in the `output` buffer.
    ///
    /// To make overlapping array segments easier, this method DOES NOT zero out the output buffer, instead it adds
    /// (via operator+) the result of the IMDCT to what's already in the buffer.
    fn process(&mut self, input: &[T], output: &mut [T]) {
        let (output_a, output_b) = output.split_at_mut(input.len());

        self.process_split(input, output_a, output_b);
    }

    /// Computes the MDCT on the `input` buffer and places the result in the `output` buffer.
    /// Puts the first half of the output in `output_a`, and puts the first half of the output in `output_b`.
    ///
    /// To make overlapping array segments easier, this method DOES NOT zero out the output buffer, instead it adds
    /// (via operator+) the result of the IMDCT to what's already in the buffer.
    fn process_split(&mut self, input: &[T], output_a: &mut [T], output_b: &mut [T]);
}

pub use self::mdct_naive::MDCTNaive;
pub use self::mdct_via_dct4::MDCTViaDCT4;

pub use self::imdct_naive::IMDCTNaive;
pub use self::imdct_via_dct4::IMDCTViaDCT4;
