use rustfft::Length;

use DCTnum;

mod dct4_via_fft;
mod dct4_via_dct3;
mod dct4_naive;

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 4 (DCT4)
pub trait DCT4<T: DCTnum>: Length {
    /// Computes the DCT Type 4 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&mut self, input: &mut [T], output: &mut [T]);
}

pub use self::dct4_via_fft::DCT4ViaFFT;
pub use self::dct4_via_dct3::DCT4ViaDCT3;
pub use self::dct4_naive::DCT4Naive;
