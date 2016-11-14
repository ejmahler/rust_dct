use std::f32;
use rustfft;
use num::{Complex, Zero, Signed, FromPrimitive};

pub struct DCT4<T> {
    fft: rustfft::FFT<T>,
    fft_input: Vec<Complex<T>>,
    fft_output: Vec<Complex<T>>,

    input_correction: Vec<Complex<T>>,
}

impl<T> DCT4<T>
    where T: Signed + FromPrimitive + Copy + 'static
{
    /// Creates a new DCT4 context that will process signals of length `len`.
    pub fn new(len: usize) -> Self {
        let fft = rustfft::FFT::new(len * 8, false);
        DCT4 {
            fft: fft,
            fft_input: vec![Zero::zero(); len * 8],
            fft_output: vec![Zero::zero(); len * 8],
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

    /// Runs the DCT4 on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

        assert_eq!(signal.len() * 8, self.fft_input.len());

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
        self.fft.process(&self.fft_input, &mut self.fft_output);

        for (index, element) in spectrum.iter_mut().enumerate() {
            *element = self.fft_output[index * 2 + 1].re * FromPrimitive::from_f32(0.25).unwrap();
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
            vec![6_f32,9_f32,1_f32,5_f32,2_f32,6_f32,2_f32,-1_f32],
        ];
        let expected_list = vec![
            vec![0_f32,0_f32,0_f32,0_f32,0_f32],
            vec![3.19623_f32, -1.10134_f32, 0.707107_f32, -0.561163_f32, 0.506233_f32],
            vec![23.9103_f32, 0.201528_f32, 5.36073_f32, 2.53127_f32, -5.21319_f32, -0.240328_f32, -9.32464_f32, -5.56147_f32],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let output = execute_slow(&input);

            compare_float_vectors(&expected.as_slice(), &output.as_slice());
        }
    }

    #[test]
    fn test_slow_inverse() {
        let input_list = vec![
            vec![0_f32,0_f32,0_f32,0_f32,0_f32],
            vec![1_f32,1_f32,1_f32,1_f32,1_f32],
            vec![6_f32,9_f32,1_f32,5_f32,2_f32,6_f32,2_f32,-1_f32],
        ];

        for input in input_list {
            let mut output = execute_slow(execute_slow(&input).as_slice());

            let scale = 2_f32 / input.len() as f32;
            for element in output.iter_mut() {
                *element = *element * scale;
            }

            compare_float_vectors(&input.as_slice(), &output.as_slice());
        }
    }


    #[test]
    fn test_fast() {
        println!("BEGINNING OF FAST $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");


        let input_list = vec![
            vec![2_f32, 0_f32],
            vec![4_f32, 0_f32, 0_f32, 0_f32],
            vec![21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
        ];

        for input in input_list {
            println!("BEGINNING OF LOOP $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");

            let slow_output = execute_slow(&input);

            let mut dct = DCT4::new(input.len());
            let mut fast_output = input.clone();
            dct.process(&input, &mut fast_output);

            println!("AFTER PROCESS ___________________________________________________________________________________");

            println!("FUCK");
            println!("observed: {:?}, expected: {:?}", fast_output, slow_output);

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }
}
