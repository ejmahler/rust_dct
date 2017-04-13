use std::rc::Rc;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use DCTnum;
use twiddles;
use dct2::DCT2;

pub struct DCT2ViaFFT<T> {
    fft: Rc<FFT<T>>,
    fft_input: Box<[Complex<T>]>,
    fft_output: Box<[Complex<T>]>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT2ViaFFT<T> {
    /// Creates a new DCT3 context that will process signals of length `inner_fft.len()`.
    pub fn new(inner_fft: Rc<FFT<T>>) -> Self {
        assert!(!inner_fft.is_inverse(), "The 'DCT type 2 via FFT' algorithm requires a forward FFT, but an inverse FFT was provided");

        let len = inner_fft.len();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| twiddles::single_twiddle(i, len * 4, false))
            .collect();

        Self {
            fft: inner_fft,
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len].into_boxed_slice(),
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len].into_boxed_slice(),
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DCTnum> DCT2<T> for DCT2ViaFFT<T> {
    fn process(&mut self, signal: &mut [T], spectrum: &mut [T]) {    

        assert!(signal.len() == self.fft_input.len());

        // the first half of the array will be the even elements, in order
        let even_end = (signal.len() + 1) / 2;
        for i in 0..even_end {
            self.fft_input[i] = Complex{ re: signal[i*2], im: T::zero() };
        }

        // the second half is the odd elements in reverse order
        let odd_end = signal.len() - 1 - signal.len() % 2;
        for i in 0..signal.len() / 2 {
            self.fft_input[even_end + i] = Complex{ re: signal[odd_end - 2 * i], im: T::zero() };
        }

        // run the fft
        self.fft.process(&mut self.fft_input, &mut self.fft_output);

        // apply a correction factor to the result
        for ((fft_entry, correction_entry),  spectrum_entry) in self.fft_output.iter().zip(self.twiddles.iter()).zip(spectrum.iter_mut()) {
            *spectrum_entry = (fft_entry * correction_entry).re;
        }
    }
}
impl<T> Length for DCT2ViaFFT<T> {
    fn len(&self) -> usize {
        self.fft_input.len()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use dct2::DCT2Naive;

    use ::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct2_via_fft() {
        for size in 2..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT2Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let mut dct = DCT2ViaFFT::new(fft_planner.plan_fft(size));
            dct.process(&mut actual_input, &mut actual_output);

            println!("{}", size);
            println!("expected: {:?}", actual_output);
            println!("actual: {:?}", actual_input);

            assert!(compare_float_vectors(&actual_output, &expected_output), "len = {}", size);
        }
    }
}
