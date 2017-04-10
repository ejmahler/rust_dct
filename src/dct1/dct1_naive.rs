use std::f64;

use rustfft::Length;

use dct1::DCT1;
use DCTnum;

pub struct DCT1Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DCTnum> DCT1Naive<T> {
    /// Creates a new DCT4 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        assert_ne!(len, 1, "DCT Type 1 is undefined for len == 1");

        let constant_factor = f64::consts::PI / ((len - 1) as f64);

        let twiddles: Vec<T> = (0..(len - 1)*2)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice()
        }
    }
}

impl<T: DCTnum> DCT1<T> for DCT1Naive<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {
        assert_eq!(input.len(), self.len());

        let half = T::from_f32(0.5f32).unwrap();
        input[0] = input[0] * half;
        input[self.len() - 1] = input[self.len() - 1] * half;

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = input[0];
            
            let twiddle_stride = k;
            let mut twiddle_index = twiddle_stride;

            for i in 1..input.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + input[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
        
    }
}
impl<T> Length for DCT1Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 2 + 1
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use num::Zero;
    use test_utils::{compare_float_vectors, random_signal};
    use std::f32;

    fn slow_dct1(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());
        let twiddle_constant = f32::consts::PI / ((input.len() - 1) as f32);

        for k in 0..input.len() {
            let mut current_value = if k %2 == 0 {
                (input[0] + input[input.len() - 1]) * 0.5f32
            } else {
                (input[0] - input[input.len() - 1]) * 0.5f32
            };

            let k_float = k as f32;

            for i in 1..(input.len() - 1) {
                let i_float = i as f32;

                let twiddle = (k_float * i_float * twiddle_constant).cos();

                current_value += input[i] * twiddle;
            }
            result.push(current_value);

        }
        return result;
    }


    #[test]
    fn test_known_lengths() {
        let input_list = vec![
            vec![1_f32, 1_f32],
            vec![1_f32, 2_f32, 3_f32, 5_f32],
            vec![1_f32, 2_f32, 3_f32, 5_f32, 1_f32, -3_f32],
        ]; 
        let expected_list = vec![
            vec![1_f32, 0_f32],
            vec![8_f32, -2.5_f32, 0.5_f32, -1_f32],
            vec![10.0_f32, 2.1909830056250525_f32, -6.5450849718747373_f32, 3.3090169943749475_f32, -0.95491502812526274_f32, -1.0_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let slow_output = slow_dct1(&input);

            let mut actual_input = input.to_vec();
            let mut actual_output = vec![Zero::zero(); input.len()];

            let mut dct = DCT1Naive::new(input.len());

            dct.process(&mut actual_input, &mut actual_output);

            println!("known: {:?}", expected);
            println!("slow:  {:?}", slow_output);
            println!("fast:  {:?}", actual_output);

            assert!(compare_float_vectors(&expected, &slow_output));
            assert!(compare_float_vectors(&expected, &actual_output));
        }
    }


    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_matches_dct1() {
        for size in 2..20 {
            let mut input = random_signal(size);
            let slow_output = slow_dct1(&input);

            
            let mut fast_output = vec![0f32; size];

            let mut dct = DCT1Naive::new(size);

            dct.process(&mut input, &mut fast_output);

            println!("{}", size);
            println!("expected: {:?}", slow_output);
            println!("actual: {:?}", fast_output);

            assert!(compare_float_vectors(&slow_output, &fast_output), "len = {}", size);
        }
    }
}