use std::f32;
use std::rc::Rc;

use rustfft::{FFT, Planner};
use num::{Complex, Zero, FromPrimitive};

use DCTnum;

pub struct DCT3<T> {
    fft: Rc<FFT<T>>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,

    input_correction: Vec<Complex<T>>,
}

impl<T: DCTnum> DCT3<T> {
    /// Creates a new DCT3 context that will process signals of length `len`.
    pub fn new(len: usize) -> Self {
        let mut planner = Planner::new(false);
        DCT3 {
            fft: planner.plan_fft(len),
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            input_correction: (0..len)
                .map(|i| i as f32 * 0.5 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&0.5, &phase).conj())
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
        }
    }

    pub fn new_with_planner(len: usize, planner: &mut Planner<T>) -> Self {
        DCT3 {
            fft: planner.plan_fft(len),
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            input_correction: (0..len)
                .map(|i| i as f32 * 0.5 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&0.5, &phase).conj())
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
        }
    }

    /// Runs the DCT3 on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

        assert!(signal.len() == self.fft_input.len());

        // compute the FFT input based on the correction factors
        for i in 0..signal.len() {
            unsafe {
                let imaginary_part = if i == 0 {
                    Zero::zero()
                } else {
                    *signal.get_unchecked(signal.len() - i)
                };
                *self.fft_input.get_unchecked_mut(i) = Complex::new(*signal.get_unchecked(i),
                                                                    imaginary_part) *
                                                       *self.input_correction.get_unchecked(i);
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


#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    use ::test_utils::{compare_float_vectors, random_signal};

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

            compare_float_vectors(&expected.as_slice(), &output.as_slice());
        }
    }

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 2..50 {
            let input = random_signal(size);

            let slow_output = execute_slow(&input);

            let mut dct = DCT3::new(size);
            let mut fast_output = vec![0f32; size];
            dct.process(&input, fast_output.as_mut_slice());

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }
}
