use rustfft::Length;

mod mdct_naive;
mod mdct_via_dct4;

pub mod window_fn;

/// An umbrella trait for algorithms which compute the Modified Discrete Cosine Transform (MDCT)
pub trait Mdct<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the MDCT on the `input` buffer and places the result in the `output` buffer.
    /// Uses `input_a` for the first half of the input, and `input_b` for the second half of the input
    ///
    /// To make overlapping array segments easier, this method DOES NOT modify the input buffer.
    ///
    /// Normalization depends on which window function was chosen when planning the mdct --
    /// each built-in window function documents whether it does normalization or not.
    fn process_mdct_with_scratch(
        &self,
        input_a: &[T],
        input_b: &[T],
        output: &mut [T],
        scratch: &mut [T],
    );

    /// Computes the IMDCT on the `input` buffer and places the result in the `output` buffer.
    /// Puts the first half of the output in `output_a`, and puts the first half of the output in `output_b`.
    ///
    /// Since the IMDCT is designed with overlapping output segments in mind, this method DOES NOT zero
    /// out the output buffer before writing like most other DCT algorithms. Instead, it sums
    /// the result of the IMDCT with what's already in the output buffer.
    ///
    /// Normalization depends on which window function was chosen when planning the mdct --
    /// each built-in window function documents whether it does normalization or not.
    fn process_imdct_with_scratch(
        &self,
        input: &[T],
        output_a: &mut [T],
        output_b: &mut [T],
        scratch: &mut [T],
    );
}

use crate::{DctNum, RequiredScratch};

pub use self::mdct_naive::MdctNaive;
pub use self::mdct_via_dct4::MdctViaDct4;
