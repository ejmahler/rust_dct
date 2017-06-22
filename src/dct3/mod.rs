use rustfft::Length;

use DCTnum;

mod dct3_via_fft;
mod dct3_naive;

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 3 (DCT3)
pub trait DCT3<T: DCTnum>: Length {
    /// Computes the DCT Type 3 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct3_via_fft::DCT3ViaFFT;
pub use self::dct3_naive::DCT3Naive;
