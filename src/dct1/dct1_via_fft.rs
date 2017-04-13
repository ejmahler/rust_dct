use std::rc::Rc;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use DCTnum;
use dct1::DCT1;

pub struct DCT1ViaFFT<T> {
    fft: Rc<FFT<T>>,
    fft_input: Box<[Complex<T>]>,
    fft_output: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT1ViaFFT<T> {
    /// Creates a new DCT1 context that will process signals of length `len`.
    pub fn new(inner_fft: Rc<FFT<T>>) -> Self {
        let inner_len = inner_fft.len();

        assert!(inner_len % 2 == 0, "For DCT1 via FFT, the inner FFT size must be even. Got {}", inner_len);

        Self {
            fft: inner_fft,
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); inner_len].into_boxed_slice(),
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); inner_len].into_boxed_slice(),
        }
    }
}

impl<T: DCTnum> DCT1<T> for DCT1ViaFFT<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {   
        assert!(input.len() == self.len());

        for (&input_val, fft_cell) in input.iter().zip(&mut self.fft_input[..input.len()]) {
            *fft_cell = Complex { re: input_val, im: T::zero() };
        }
        for (&input_val, fft_cell) in input.iter().rev().skip(1).zip(&mut self.fft_input[input.len()..]) {
            *fft_cell = Complex { re: input_val, im: T::zero() };
        }

        // run the fft
        self.fft.process(&mut self.fft_input, &mut self.fft_output);

        // apply a correction factor to the result
        let half = T::from_f32(0.5f32).unwrap();
        for (fft_entry, output_val) in self.fft_output.iter().zip(output.iter_mut()) {
            *output_val = fft_entry.re * half;
        }
    }
}
impl<T> Length for DCT1ViaFFT<T> {
    fn len(&self) -> usize {
        self.fft_input.len() / 2 + 1
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use dct1::DCT1Naive;

    use ::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct1_via_fft() {
        for size in 2..20 {

            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT1Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let mut dct = DCT1ViaFFT::new(fft_planner.plan_fft((size - 1) * 2));
            dct.process(&mut actual_input, &mut actual_output);

            assert!(compare_float_vectors(&actual_output, &expected_output), "len = {}", size);
        }
    }
}
