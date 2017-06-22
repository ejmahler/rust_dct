use rustfft::Length;

use DCTnum;

mod dct1_naive;
mod dct1_via_fft;

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 1 (DCT1)
pub trait DCT1<T: DCTnum>: Length {
    /// Computes the DCT Type 1 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct1_naive::DCT1Naive;
pub use self::dct1_via_fft::DCT1ViaFFT;
