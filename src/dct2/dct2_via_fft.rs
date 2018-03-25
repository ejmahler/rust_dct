use std::sync::Arc;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use common;
use twiddles;
use dct2::DCT2;

/// DCT Type 2 implementation that converts the problem into an O(nlogn) FFT of the same size
///
/// ~~~
/// // Computes a DCT Type 2 of size 1234
/// use rustdct::dct2::{DCT2, DCT2ViaFFT};
/// use rustdct::rustfft::FFTplanner;
///
/// let len = 1234;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(len);
///
/// let dct = DCT2ViaFFT::new(fft);
/// dct.process_dct2(&mut input, &mut output);
/// ~~~
pub struct DCT2ViaFFT<T> {
    fft: Arc<FFT<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: common::DCTnum> DCT2ViaFFT<T> {
    /// Creates a new DCT2 context that will process signals of length `inner_fft.len()`.
    pub fn new(inner_fft: Arc<FFT<T>>) -> Self {
        assert!(
            !inner_fft.is_inverse(),
            "The 'DCT type 2 via FFT' algorithm requires a forward FFT, but an inverse FFT \
                 was provided"
        );

        let len = inner_fft.len();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| twiddles::single_twiddle(i, len * 4, false))
            .collect();

        Self {
            fft: inner_fft,
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: common::DCTnum> DCT2<T> for DCT2ViaFFT<T> {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let mut buffer = vec![Complex::zero(); self.len() * 2];
        let (fft_input, fft_output) = buffer.split_at_mut(self.len());

        // the first half of the array will be the even elements, in order
        let even_end = (input.len() + 1) / 2;
        for i in 0..even_end {
            fft_input[i] = Complex {
                re: input[i * 2],
                im: T::zero(),
            };
        }

        // the second half is the odd elements in reverse order
        let odd_end = input.len() - 1 - input.len() % 2;
        for i in 0..input.len() / 2 {
            fft_input[even_end + i] = Complex {
                re: input[odd_end - 2 * i],
                im: T::zero(),
            };
        }

        // run the fft
        self.fft.process(fft_input, fft_output);

        // apply a correction factor to the result
        for ((fft_entry, correction_entry), spectrum_entry) in
            fft_output.iter().zip(self.twiddles.iter()).zip(output.iter_mut())
        {
            *spectrum_entry = (fft_entry * correction_entry).re;
        }
    }
}
impl<T> Length for DCT2ViaFFT<T> {
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use dct2::DCT2Naive;

    use test_utils::{compare_float_vectors, random_signal};
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct2_via_fft() {
        for size in 2..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let naive_dct = DCT2Naive::new(size);
            naive_dct.process_dct2(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let dct = DCT2ViaFFT::new(fft_planner.plan_fft(size));
            dct.process_dct2(&mut actual_input, &mut actual_output);

            println!("{}", size);
            println!("expected: {:?}", actual_output);
            println!("actual: {:?}", actual_input);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }
}
