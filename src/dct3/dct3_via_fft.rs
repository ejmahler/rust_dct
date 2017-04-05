use std::rc::Rc;

use num::{Complex, Zero, FromPrimitive};
use rustfft::{FFT, Length};

use DCTnum;
use twiddles;
use dct3::DCT3;

pub struct DCT3ViaFFT<T> {
    fft: Rc<FFT<T>>,
    fft_input: Box<[Complex<T>]>,
    fft_output: Box<[Complex<T>]>,

    twiddles: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT3ViaFFT<T> {
    /// Creates a new DCT3 context that will process signals of length `len`.
    pub fn new(inner_fft: Rc<FFT<T>>) -> Self {

        let len = inner_fft.len();

        let half = T::from_f32(0.5f32).unwrap();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| twiddles::single_twiddle(i, len * 4, false) * half)
            .collect();

        Self {
            fft: inner_fft,
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len].into_boxed_slice(),
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len].into_boxed_slice(),
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DCTnum> DCT3<T> for DCT3ViaFFT<T> {
    fn process(&mut self, signal: &mut [T], spectrum: &mut [T]) {    

        assert!(signal.len() == self.fft_input.len());

        // compute the FFT input based on the correction factors
        for i in 0..signal.len() {
            unsafe {
                let imaginary_part = if i == 0 {
                    T::zero()
                } else {
                    *signal.get_unchecked(signal.len() - i)
                };
                *self.fft_input.get_unchecked_mut(i) = Complex::new(*signal.get_unchecked(i),
                                                                    imaginary_part) *
                                                       *self.twiddles.get_unchecked(i);
            }
        }

        // run the fft
        self.fft.process(&mut self.fft_input, &mut self.fft_output);

        // copy the first half of the fft output into the even elements of the spectrum
        let even_end = (signal.len() + 1) / 2;
        for i in 0..even_end {
            unsafe {
                *spectrum.get_unchecked_mut(i * 2) = (*self.fft_output.get_unchecked(i)).re;
            }
        }

        // copy the second half of the fft output into the odd elements, reversed
        let odd_end = signal.len() - 1 - signal.len() % 2;
        for i in 0..signal.len() / 2 {
            unsafe {
                *spectrum.get_unchecked_mut(odd_end - 2 * i) =
                    (*self.fft_output.get_unchecked(i + even_end)).re;
            }
        }
    }
}
impl<T> Length for DCT3ViaFFT<T> {
    fn len(&self) -> usize {
        self.fft_input.len()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    use ::test_utils::{compare_float_vectors, random_signal};
    use rustfft::Planner;

    fn execute_slow(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());

        let size_float = input.len() as f32;

        for k in 0..input.len() {
            let mut current_value = input[0] * 0.5_f32;

            let k_float = k as f32;

            for i in 1..(input.len()) {
                let i_float = i as f32;

                current_value +=
                    input[i] * (f32::consts::PI * i_float * (k_float + 0.5_f32) / size_float).cos();
            }
            result.push(current_value);

        }

        return result;
    }


    #[test]
    fn test_slow() {
        let input_list = vec![
            vec![2_f32, 0_f32],
            vec![4_f32, 0_f32, 0_f32, 0_f32],
            vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
        ];
        let expected_list = vec![
            vec![1_f32, 1_f32],
            vec![2_f32, 2_f32, 2_f32, 2_f32],
            vec![10_f32, 2.5_f32, 15_f32, 5_f32, 20_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let output = execute_slow(&input);

            println!("{:?}", output);

            compare_float_vectors(&expected, &output);
        }
    }

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 2..25 {
            let mut input = random_signal(size);

            let slow_output = execute_slow(&input);

            let mut planner = Planner::new(false);
            let inner_fft = planner.plan_fft(size);

            let mut dct = DCT3ViaFFT::new(inner_fft);
            let mut fast_output = vec![0f32; size];
            dct.process(&mut input, &mut fast_output);

            compare_float_vectors(&slow_output, &fast_output);
        }
    }
}
