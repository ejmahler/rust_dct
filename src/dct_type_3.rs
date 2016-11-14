use std::f32;
use rustfft;
use num::{Complex, Zero, Signed, FromPrimitive};

use super::math_utils;

pub struct DCT3<T> {
    fft: rustfft::FFT<T>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,

    input_correction: Vec<Complex<T>>,
}

impl<T> DCT3<T>
    where T: Signed + FromPrimitive + Copy + 'static
{
    /// Creates a new DCT3 context that will process signals of length `len`.
    pub fn new(len: usize) -> Self {
        let fft = rustfft::FFT::new(len, false);
        DCT3 {
            fft: fft,
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
        self.fft.process(&self.fft_input, &mut self.fft_output);

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

// perform a 2-dimensional dct3 on the input data, putting the result into the input vector
pub fn dct3_2d(width: usize, height: usize, row_major_data: &mut [f32]) {
    let mut intermediate = vec![0_f32; row_major_data.len()];

    // transpose the data into the intermediate vector
    math_utils::transpose(width, height, row_major_data, intermediate.as_mut_slice());

    // perform DCTs down the columns
    {
        let mut height_dct = DCT3::new(height);
        for (input, output) in intermediate.chunks(height).zip(row_major_data.chunks_mut(height)) {
            height_dct.process(input, output);
        }
    }

    // transpose the data into the intermediate vector again
    math_utils::transpose(height, width, row_major_data, intermediate.as_mut_slice());

    // perform DCTs down the columns
    {
        let mut width_dct = DCT3::new(width);
        for (input, output) in intermediate.chunks(width).zip(row_major_data.chunks_mut(width)) {
            width_dct.process(input, output);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
        a >= b - tolerance && a <= b + tolerance
    }

    fn compare_float_vectors(expected: &[f32], observed: &[f32]) {
        assert_eq!(expected.len(), observed.len());

        let tolerance: f32 = 0.0001;

        for i in 0..expected.len() {
            assert!(fuzzy_cmp(observed[i], expected[i], tolerance));
        }
    }


    pub fn execute_slow(input: &[f32]) -> Vec<f32> {
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

    #[test]
    fn test_fast() {
        let input_list = vec![
            vec![2_f32, 0_f32],
            vec![4_f32, 0_f32, 0_f32, 0_f32],
            vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
        ];

        for input in input_list {
            let slow_output = execute_slow(&input);

            let mut dct = DCT3::new(input.len());
            let mut fast_output = input.clone();
            dct.process(&input, &mut fast_output);

            println!("{:?}", slow_output);
            println!("{:?}", fast_output);

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }

    #[test]
    fn test_2d() {
        let input_list = vec![
            (2 as usize, 2 as usize, vec![
                4_f32, 0_f32,
                0_f32, 0_f32,
            ]),
            (3 as usize, 2 as usize, vec![
                6_f32, 0_f32, 0_f32,
                0_f32, 0_f32, 0_f32,
            ]),
        ];
        let expected_list = vec![
            vec![
                1_f32, 1_f32,
                1_f32, 1_f32,
            ],
            vec![
                1.5_f32, 1.5_f32, 1.5_f32,
                1.5_f32, 1.5_f32, 1.5_f32,
            ],
        ];

        for i in 0..input_list.len() {
            let (width, height, ref input) = input_list[i];

            let mut output = input.clone();
            dct3_2d(width, height, output.as_mut_slice());

            println!("");
            println!("input:   {:?}", input);
            println!("actual:  {:?}", output);
            println!("expected:{:?}", expected_list[i]);

            compare_float_vectors(&expected_list[i].as_slice(), &output.as_slice());
        }
    }
}
