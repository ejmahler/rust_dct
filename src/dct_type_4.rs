use std::rc::Rc;

use rustfft;
use num::{Complex, Zero, FromPrimitive};

pub struct DCT4<T> {
    fft: Rc<rustfft::FFT<T>>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,
}

impl<T: rustfft::FFTnum> DCT4<T> {
    /// Creates a new DCT4 context that will process signals of length `len`.
    pub fn new(len: usize) -> Self {
        let mut planner = rustfft::Planner::new(false);
        DCT4 {
            fft: planner.plan_fft(len * 8),
            fft_input: vec![Zero::zero(); len * 8],
            fft_output: vec![Zero::zero(); len * 8],
        }
    }

    pub fn new_with_planner(len: usize, planner: &mut rustfft::Planner<T>) -> Self {
        DCT4 {
            fft: planner.plan_fft(len * 8),
            fft_input: vec![Zero::zero(); len * 8],
            fft_output: vec![Zero::zero(); len * 8],
        }
    }

    /// Runs the DCT4 on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

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
        self.fft.process(&mut self.fft_input, &mut self.fft_output);

        for (index, element) in spectrum.iter_mut().enumerate() {
            *element = self.fft_output[index * 2 + 1].re * FromPrimitive::from_f32(0.25).unwrap();
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ::test_utils::{compare_float_vectors, random_signal};
    use std::f32;

    fn execute_slow(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());

        let size_float = input.len() as f32;

        for k in 0..input.len() {
            let mut current_value = 0_f32;

            let k_float = k as f32;

            for i in 0..input.len() {
                let i_float = i as f32;

                current_value +=
                    input[i] * (f32::consts::PI * (i_float + 0.5_f32) * (k_float + 0.5_f32) / size_float).cos();
            }
            result.push(current_value);

        }

        return result;
    }


    #[test]
    fn test_slow() {
        let input_list = vec![
            vec![0_f32,0_f32,0_f32,0_f32,0_f32],
            vec![1_f32,1_f32,1_f32,1_f32,1_f32],
            vec![4.7015433_f32, -11.926178_f32, 27.098675_f32, -1.9793236_f32],
            vec![6_f32,9_f32,1_f32,5_f32,2_f32,6_f32,2_f32,-1_f32],
        ];
        let expected_list = vec![
            vec![0_f32,0_f32,0_f32,0_f32,0_f32],
            vec![3.19623_f32, -1.10134_f32, 0.707107_f32, -0.561163_f32, 0.506233_f32],
            vec![9.36402_f32, -19.242455_f32, 17.949997_f32, 32.01607_f32],
            vec![23.9103_f32, 0.201528_f32, 5.36073_f32, 2.53127_f32, -5.21319_f32, -0.240328_f32, -9.32464_f32, -5.56147_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let output = execute_slow(&input);

            compare_float_vectors(&expected.as_slice(), &output.as_slice());
        }
    }


    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for size in 1..50 {
            let input = random_signal(size);

            let slow_output = execute_slow(&input);

            let mut dct = DCT4::new(size);
            let mut fast_output = vec![0f32; size];
            dct.process(&input, fast_output.as_mut_slice());

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }
}
