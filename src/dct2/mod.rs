use rustfft::Length;

use DCTnum;

mod dct2_via_fft;
mod dct2_splitradix;
mod dct2_naive;

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 2 (DCT2)
pub trait DCT2<T: DCTnum>: Length {
    /// Computes the DCT Type 2 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct2_via_fft::DCT2ViaFFT;
pub use self::dct2_splitradix::DCT2SplitRadix;
pub use self::dct2_naive::DCT2Naive;
