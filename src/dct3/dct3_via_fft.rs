use std::sync::Arc;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use common;
use twiddles;
use dct3::DCT3;

/// DCT Type 3 implementation that converts the problem into an O(nlogn) FFT of the same size
///
/// ~~~
/// // Computes a DCT Type 3 of size 1234
/// use rustdct::dct3::{DCT3, DCT3ViaFFT};
/// use rustdct::rustfft::FFTplanner;
///
/// let len = 1234;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(len);
///
/// let dct = DCT3ViaFFT::new(fft);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT3ViaFFT<T> {
    fft: Arc<FFT<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: common::DCTnum> DCT3ViaFFT<T> {
    /// Creates a new DCT3 context that will process signals of length `inner_fft.len()`.
    pub fn new(inner_fft: Arc<FFT<T>>) -> Self {
        assert!(
            !inner_fft.is_inverse(),
            "The 'DCT type 3 via FFT' algorithm requires a forward FFT, but an inverse FFT \
                 was provided"
        );

        let len = inner_fft.len();

        let half = T::from_f32(0.5f32).unwrap();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| twiddles::single_twiddle(i, len * 4, false) * half)
            .collect();

        Self {
            fft: inner_fft,
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: common::DCTnum> DCT3<T> for DCT3ViaFFT<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let mut buffer = vec![Complex::zero(); self.len() * 2];
        let (mut fft_input, mut fft_output) = buffer.split_at_mut(self.len());

        // compute the FFT input based on the correction factors
        for i in 0..input.len() {
            let imaginary_part = if i == 0 {
                T::zero()
            } else {
                input[input.len() - i]
            };
            let c = Complex {
                re: input[i],
                im: imaginary_part,
            };

            fft_input[i] = c * self.twiddles[i];
        }

        // run the fft
        self.fft.process(&mut fft_input, &mut fft_output);

        // copy the first half of the fft output into the even elements of the output
        let even_end = (input.len() + 1) / 2;
        for i in 0..even_end {
            output[i * 2] = fft_output[i].re;
        }

        // copy the second half of the fft output into the odd elements, reversed
        let odd_end = input.len() - 1 - input.len() % 2;
        for i in 0..input.len() / 2 {
            output[odd_end - 2 * i] = fft_output[i + even_end].re;
        }
    }
}
impl<T> Length for DCT3ViaFFT<T> {
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use dct3::DCT3Naive;

    use test_utils::{compare_float_vectors, random_signal};
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct3_via_fft() {
        for size in 2..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT3Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let mut dct = DCT3ViaFFT::new(fft_planner.plan_fft(size));
            dct.process(&mut actual_input, &mut actual_output);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }
}
