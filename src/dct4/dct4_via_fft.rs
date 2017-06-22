use std::sync::Arc;
use std::f64;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use twiddles;
use DCTnum;
use dct4::DCT4;

/// DCT Type 4 implementation that converts the problem into an O(nlogn) FFT of size N * 4
///
/// ~~~
/// // Computes a DCT Type 4 of size 1234
/// use rustdct::dct4::{DCT4, DCT4ViaFFT};
/// use rustdct::rustfft::FFTplanner;
///
/// let mut input:  Vec<f32> = vec![0f32; 1234];
/// let mut output: Vec<f32> = vec![0f32; 1234];
///
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(1234 * 4);
///
/// let mut dct = DCT4ViaFFT::new(fft);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT4ViaFFT<T> {
    inner_fft: Arc<FFT<T>>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT4ViaFFT<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len() / 4`.
    pub fn new(inner_fft: Arc<FFT<T>>) -> Self {
        let inner_len = inner_fft.len();
        assert_eq!(
            inner_len % 4,
            0,
            "inner_fft.len() for DCT4ViaFFT must be a multiple of 4. The DCT4 length will \
                    be inner_fft.len() / 4. Got {}",
            inner_fft.len()
        );
        assert!(
            !inner_fft.is_inverse(),
            "The 'DCT type 4 via FFT' algorithm requires a forward FFT, but an inverse FFT \
                 was provided"
        );

        let len = inner_fft.len() / 4;

        let twiddle_scale = T::from_f64(0.25f64 / (f64::consts::PI / (inner_len as f64)).cos())
            .unwrap();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| {
                twiddles::single_twiddle(i, inner_len, false) * twiddle_scale
            })
            .collect();

        Self {
            inner_fft: inner_fft,
            fft_input: vec![Zero::zero(); inner_len],
            fft_output: vec![Zero::zero(); inner_len],
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}
impl<T: DCTnum> DCT4<T> for DCT4ViaFFT<T> {
    fn process(&mut self, signal: &mut [T], spectrum: &mut [T]) {
        assert_eq!(signal.len(), self.len());

        //the odd elements are the DCT input, repeated and reversed and etc
        for (fft_element, input_element) in self.fft_input.iter_mut().zip(signal.iter()) {
            *fft_element = Complex::from(*input_element);
        }
        for (fft_element, input_element) in
            self.fft_input.iter_mut().skip(signal.len()).zip(
                signal
                    .iter()
                    .rev(),
            )
        {
            *fft_element = Complex::from(-*input_element);
        }
        for (fft_element, input_element) in
            self.fft_input.iter_mut().skip(signal.len() * 2).zip(
                signal
                    .iter(),
            )
        {
            *fft_element = Complex::from(-*input_element);
        }
        for (fft_element, input_element) in
            self.fft_input.iter_mut().skip(signal.len() * 3).zip(
                signal
                    .iter()
                    .rev(),
            )
        {
            *fft_element = Complex::from(*input_element);
        }

        // run the fft
        self.inner_fft.process(
            &mut self.fft_input,
            &mut self.fft_output,
        );

        for (index, (element, twiddle)) in
            spectrum.iter_mut().zip(self.twiddles.iter()).enumerate()
        {
            *element = (self.fft_output[index * 2 + 1] * twiddle).re;
        }
    }
}
impl<T> Length for DCT4ViaFFT<T> {
    fn len(&self) -> usize {
        self.fft_input.len() / 4
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use dct4::DCT4Naive;
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 1..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT4Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let mut dct = DCT4ViaFFT::new(fft_planner.plan_fft(size * 4));
            dct.process(&mut actual_input, &mut actual_output);

            let divided: Vec<f32> = expected_output
                .iter()
                .zip(actual_output.iter())
                .map(|(&a, b)| b / a)
                .collect();

            println!("");
            println!("expected: {:?}", expected_output);
            println!("actual:   {:?}", actual_output);
            println!("divided:  {:?}", divided);

            assert!(
                compare_float_vectors(&expected_output, &actual_output),
                "len = {}",
                size
            );
        }
    }
}
