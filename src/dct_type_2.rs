use std::f32;
use std::rc::Rc;

use num::{Complex, Zero, FromPrimitive};
use rustfft::{FFT, Planner};

use DCTnum;

pub struct DCT2<T> {
    fft: Rc<FFT<T>>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,

    output_correction: Vec<Complex<T>>,
}

impl<T: DCTnum> DCT2<T> {
    /// Creates a new DCT2 context that will process signals of length `len`.
    pub fn new(len: usize) -> Self {
        let mut planner = Planner::new(false);
        DCT2 {
            fft: planner.plan_fft(len),
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            output_correction: (0..len)
                .map(|i| i as f32 * 0.5 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase).conj())
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
        }
    }

    /// Creates a new DCT2 context that will process signals of length `len`.
    pub fn new_with_planner(len: usize, planner: &mut Planner<T>) -> Self {
        DCT2 {
            fft: planner.plan_fft(len),
            fft_input: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            fft_output: vec![Complex::new(Zero::zero(),Zero::zero()); len],
            output_correction: (0..len)
                .map(|i| i as f32 * 0.5 * f32::consts::PI / len as f32)
                .map(|phase| Complex::from_polar(&1.0, &phase).conj())
                .map(|c| {
                    Complex {
                        re: FromPrimitive::from_f32(c.re).unwrap(),
                        im: FromPrimitive::from_f32(c.im).unwrap(),
                    }
                })
                .collect(),
        }
    }

    /// Runs the DCT2 on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

        // we're going to convert this to a FFT. we'll do so by redordering the inputs,
        // running the FFT, and then multiplying by a correction factor
        assert!(signal.len() == self.fft_input.len());

        // the first half of the array will be the even elements, in order
        let even_end = (signal.len() + 1) / 2;
        for i in 0..even_end {
            unsafe {
                *self.fft_input.get_unchecked_mut(i) = Complex::from(*signal.get_unchecked(i * 2));
            }
        }

        // the second half is the odd elements in reverse order
        let odd_end = signal.len() - 1 - signal.len() % 2;
        for i in 0..signal.len() / 2 {
            unsafe {
                *self.fft_input.get_unchecked_mut(even_end + i) =
                    Complex::from(*signal.get_unchecked(odd_end - 2 * i));
            }
        }

        // run the fft
        self.fft.process(&mut self.fft_input, &mut self.fft_output);

        // apply a correction factor to the result
        for ((fft_entry, correction_entry),  spectrum_entry) in self.fft_output.iter().zip(self.output_correction.iter()).zip(spectrum.iter_mut()) {
            *spectrum_entry = (fft_entry * correction_entry).re;
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
            let mut current_value = 0_f32;

            let k_float = k as f32;

            for i in 0..(input.len()) {
                let i_float = i as f32;

                current_value +=
                    input[i] * (f32::consts::PI * k_float * (i_float + 0.5_f32) / size_float).cos();
            }
            result.push(current_value);

        }

        return result;
    }


    #[test]
    fn test_slow() {
        let input_list = vec![
            vec![1_f32, 1_f32],
            vec![1_f32, 1_f32, 1_f32, 1_f32],
            vec![4_f32, 1_f32, 6_f32, 2_f32, 8_f32],
        ];
        let expected_list = vec![
            vec![2_f32, 0_f32],
            vec![4_f32, 0_f32, 0_f32, 0_f32],
            vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let output = execute_slow(&input);

            println!("{:?}", output);

            compare_float_vectors(&expected.as_slice(), &output.as_slice());
        }
    }

    /// Verify that our fast implementation of the DCT2 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 2..50 {
            let input = random_signal(size);

            let slow_output = execute_slow(&input);

            let mut dct = DCT2::new(size);
            let mut fast_output = vec![0f32; size];
            dct.process(&input, fast_output.as_mut_slice());

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }
}
