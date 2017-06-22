use std::f64;

use rustfft::Length;

use dct3::DCT3;
use DCTnum;

/// Naive O(n^2 ) DCT Type 3 implementation
///
/// This implementation is primarily used to test other DCT3 algorithms. In rare cases, this algorithm may be faster
/// than `DCT3ViaFFT`.
///
/// ~~~
/// // Computes a naive DCT3 of size 123
/// use rustdct::dct3::{DCT3, DCT3Naive};
///
/// let mut input:  Vec<f32> = vec![0f32; 123];
/// let mut output: Vec<f32> = vec![0f32; 123];
///
/// let mut dct = DCT3Naive::new(123);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT3Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DCTnum> DCT3Naive<T> {
    /// Creates a new DCT3 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len * 4)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: DCTnum> DCT3<T> for DCT3Naive<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {
        assert_eq!(input.len(), self.len());

        let half_first = T::from_f32(0.5f32).unwrap() * input[0];

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = half_first;

            let twiddle_stride = k * 2 + 1;
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
impl<T> Length for DCT3Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use std::f32;

    fn slow_dct3(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());

        let size_float = input.len() as f32;

        for k in 0..input.len() {
            let mut current_value = input[0] * 0.5_f32;

            let k_float = k as f32;

            for i in 1..(input.len()) {
                let i_float = i as f32;

                let twiddle = (f32::consts::PI * i_float * (k_float + 0.5_f32) / size_float).cos();

                current_value += input[i] * twiddle;
            }
            result.push(current_value);

        }

        return result;
    }


    #[test]
    fn test_known_lengths() {
        let input_list =
            vec![vec![2_f32, 0_f32],
                 vec![4_f32, 0_f32, 0_f32, 0_f32],
                 vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32]];
        let expected_list = vec![vec![1_f32, 1_f32],
                                 vec![2_f32, 2_f32, 2_f32, 2_f32],
                                 vec![10_f32, 2.5_f32, 15_f32, 5_f32, 20_f32]];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let slow_output = slow_dct3(&input);

            let mut actual_input = input.to_vec();
            let mut actual_output = vec![0f32; input.len()];

            let mut dct = DCT3Naive::new(input.len());

            dct.process(&mut actual_input, &mut actual_output);

            assert!(compare_float_vectors(&expected, &slow_output));
            assert!(compare_float_vectors(&expected, &actual_output));
        }
    }


    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_matches_dct3() {
        for size in 1..20 {
            let mut input = random_signal(size);
            let slow_output = slow_dct3(&input);


            let mut fast_output = vec![0f32; size];

            let mut dct = DCT3Naive::new(size);

            dct.process(&mut input, &mut fast_output);

            println!("{}", size);
            println!("expected: {:?}", slow_output);
            println!("actual: {:?}", fast_output);

            assert!(compare_float_vectors(&slow_output, &fast_output),
                    "len = {}",
                    size);
        }
    }
}
