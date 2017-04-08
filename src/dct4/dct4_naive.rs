use std::f64;

use num::{Zero, FromPrimitive};
use rustfft::Length;

use dct4::DCT4;
use DCTnum;

pub struct DCT4Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DCTnum> DCT4Naive<T> {
    /// Creates a new DCT4 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len*4)
            .map(|i| (constant_factor * (i as f64 + 0.5_f64)).cos())
            .map(|c| FromPrimitive::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice()
        }
    }
}

impl<T: DCTnum> DCT4<T> for DCT4Naive<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {
        assert_eq!(input.len(), self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = Zero::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

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
impl<T> Length for DCT4Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use std::f32;

    fn slow_dct4(input: &[f32]) -> Vec<f32> {
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
    fn test_known_lengths() {
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
            let slow_output = slow_dct4(&input);

            let mut actual_input = input.to_vec();
            let mut actual_output = vec![Zero::zero(); input.len()];

            let mut dct = DCT4Naive::new(input.len());

            dct.process(&mut actual_input, &mut actual_output);

            assert!(compare_float_vectors(&expected, &slow_output));
            assert!(compare_float_vectors(&expected, &actual_output));
        }
    }


    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_matches_dct4() {
        for size in 1..20 {
            let mut input = random_signal(size);
            let slow_output = slow_dct4(&input);

            
            let mut fast_output = vec![0f32; size];

            let mut dct = DCT4Naive::new(size);

            dct.process(&mut input, &mut fast_output);

            println!("{}", size);
            println!("expected: {:?}", slow_output);
            println!("actual: {:?}", fast_output);

            assert!(compare_float_vectors(&slow_output, &fast_output), "len = {}", size);
        }
    }
}
