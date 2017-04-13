use std::f64;

use rustfft::Length;

use dct2::DCT2;
use DCTnum;

pub struct DCT2Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DCTnum> DCT2Naive<T> {
    /// Creates a new DCT4 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len*4)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice()
        }
    }
}

impl<T: DCTnum> DCT2<T> for DCT2Naive<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {
        assert_eq!(input.len(), self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();
            
            let twiddle_stride = k * 2;
            let mut twiddle_index = k;

            for i in 0..input.len() {
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
impl<T> Length for DCT2Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use std::f32;

    fn slow_dct2(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());
        let size_float = input.len() as f32;

        for k in 0..input.len() {
            let mut current_value = 0_f32;

            let k_float = k as f32;

            for i in 0..(input.len()) {
                let i_float = i as f32;

                let twiddle = (f32::consts::PI * k_float * (i_float + 0.5_f32) / size_float).cos();

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
            vec![1_f32, 1_f32, 1_f32, 1_f32],
            vec![4_f32, 1_f32, 6_f32, 2_f32, 8_f32],
        ];
        let expected_list = vec![
            vec![2_f32, 0_f32],
            vec![4_f32, 0_f32, 0_f32, 0_f32],
            vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let slow_output = slow_dct2(&input);

            let mut actual_input = input.to_vec();
            let mut actual_output = vec![0f32; input.len()];

            let mut dct = DCT2Naive::new(input.len());

            dct.process(&mut actual_input, &mut actual_output);

            assert!(compare_float_vectors(&expected, &slow_output));
            assert!(compare_float_vectors(&expected, &actual_output));
        }
    }


    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_matches_dct2() {
        for size in 1..20 {
            let mut input = random_signal(size);
            let slow_output = slow_dct2(&input);

            
            let mut fast_output = vec![0f32; size];

            let mut dct = DCT2Naive::new(size);

            dct.process(&mut input, &mut fast_output);

            println!("{}", size);
            println!("expected: {:?}", slow_output);
            println!("actual: {:?}", fast_output);

            assert!(compare_float_vectors(&slow_output, &fast_output), "len = {}", size);;
        }
    }
}
