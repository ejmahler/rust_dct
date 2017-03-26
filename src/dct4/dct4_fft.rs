use std::rc::Rc;

use num::{Complex, Zero, FromPrimitive};
use rustfft::{FFT, Length};

use DCTnum;
use dct4::DCT4;

pub struct DCT4ViaFFT<T> {
    inner_fft: Rc<FFT<T>>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,
}

impl<T: DCTnum> DCT4ViaFFT<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len() / 8`.
    pub fn new(inner_fft: Rc<FFT<T>>) -> Self {
        let inner_len = inner_fft.len();
        assert_eq!(inner_len % 8, 0, "inner_fft.len() for DCT4ViaFFT must be a multiple of 8. The DCT4 length will be inner_fft.len() / 8. Got {}", inner_fft.len());

        Self {
            inner_fft: inner_fft,
            fft_input: vec![Zero::zero(); inner_len],
            fft_output: vec![Zero::zero(); inner_len],
        }
    }
}
impl<T: DCTnum> DCT4<T> for DCT4ViaFFT<T> {
    fn process(&mut self, signal: &[T], spectrum: &mut [T]) {
        assert_eq!(signal.len() * 8, self.fft_input.len());

        //all even elements are zero
        for i in 0..self.fft_input.len() / 2 {
            self.fft_input[i * 2] = Zero::zero();
        }

        //the odd elements are the DCT input, repeated and reversed and etc
        for (index, element) in signal.iter().enumerate() {
            self.fft_input[index * 2 + 1] = Complex::from(*element);
        }
        for (index, element) in signal.iter().rev().enumerate() {
            self.fft_input[signal.len() * 2 + index * 2 + 1] = Complex::from(-*element);
        }
        for (index, element) in signal.iter().enumerate() {
            self.fft_input[signal.len() * 4 + index * 2 + 1] = Complex::from(-*element);
        }
        for (index, element) in signal.iter().rev().enumerate() {
            self.fft_input[signal.len() * 6 + index * 2 + 1] = Complex::from(*element);
        }

        // run the fft
        self.inner_fft.process(&mut self.fft_input, &mut self.fft_output);

        for (index, element) in spectrum.iter_mut().enumerate() {
            *element = self.fft_output[index * 2 + 1].re * FromPrimitive::from_f32(0.25).unwrap();
        }
    }
}
impl<T> Length for DCT4ViaFFT<T> {
    fn len(&self) -> usize {
        self.fft_input.len() / 8
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use dct4::DCT4Naive;
    use rustfft::Planner;

    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 1..20 {
            let input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT4Naive::new(size);
            naive_dct.process(&input, &mut expected_output);

            let mut fft_planner = Planner::new(false);
            let mut dct = DCT4ViaFFT::new(fft_planner.plan_fft(size * 8));
            dct.process(&input, &mut actual_output);

            compare_float_vectors(&expected_output, &actual_output);
        }
    }
}
