use rustfft::num_complex::Complex;
use rustfft::Length;

use DCTnum;
use twiddles;
use dct2::DCT2;

use dct2::DCT2Naive;

/// DCT Type 2 implemention that recursively divides the problem in half. The problem size must be a power of two.
///
/// ~~~
/// // Computes a DCT Type 2 of size 1024
/// use rustdct::dct2::{DCT2, DCT2SplitRadix};
///
/// let mut input:  Vec<f32> = vec![0f32; 1024];
/// let mut output: Vec<f32> = vec![0f32; 1024];
///
/// let mut dct = DCT2SplitRadix::new(1024);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT2SplitRadix<T> {
    twiddles: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT2SplitRadix<T> {
    /// Creates a new DCT2 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        assert!(
            len.is_power_of_two() && len > 2,
            "The DCT2SplitRadix algorithm requires a power-of-two input size greater than two. Got {}", len 
        );

        let twiddles: Vec<Complex<T>> = (0..(len/2))
            .map(|i| twiddles::single_twiddle(i, len * 4, true))
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }

    fn process_recursive(&self, input: &mut [T], output: &mut [T], twiddle_stride: usize) {
        match input.len() {
            1 => output[0] = input[0],
            2 => {
                output[0] = input[0] + input[1];
                output[1] = (input[0] - input[1]) * T::FRAC_1_SQRT_2();
            },
            _ => self.process_step(input, output, twiddle_stride),
        }
    }

    fn process_step(&self, input: &mut [T], output: &mut [T], twiddle_stride: usize) {
        let len = input.len();
        let half_len = input.len() / 2;
        let quarter_len = len / 4;

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        {
            let (mut input_dct2, mut input_dct4) = output.split_at_mut(half_len);
            let (mut input_dct4_even, mut input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

            for i in 0..quarter_len {
                input_dct2[i] =                 input[len - i - 1] + input[i];
                input_dct2[half_len - i - 1] =  input[half_len - i - 1] + input[half_len + i];


                let lower_dct4 = input[i] - input[len - i - 1];
                let upper_dct4 = input[half_len - i - 1] - input[half_len + i];

                let twiddle = self.twiddles[(2 * i + 1) * twiddle_stride];

                input_dct4_even[i] = lower_dct4 * twiddle.re + upper_dct4 * twiddle.im;
                let sin_input =      upper_dct4 * twiddle.re - lower_dct4 * twiddle.im;

                input_dct4_odd[quarter_len - i - 1] = if i % 2 == 0 {
                    sin_input
                } else {
                    -sin_input
                };
            }

            // compute the recursive DCT2s
            let (mut output_dct2, mut output_dct4) = input.split_at_mut(half_len);
            let (mut output_dct4_even, mut output_dct4_odd) = output_dct4.split_at_mut(quarter_len);

            self.process_recursive(&mut input_dct2, &mut output_dct2, twiddle_stride * 2);
            self.process_recursive(&mut input_dct4_even, &mut output_dct4_even, twiddle_stride * 4);
            self.process_recursive(&mut input_dct4_odd, &mut output_dct4_odd, twiddle_stride * 4);
        }

        let (output_dct2, output_dct4) = input.split_at(half_len);
        let (output_dct4_even, output_dct4_odd) = output_dct4.split_at(quarter_len);


        //post process the 3 DCT3 outputs/ the first few and the last will be done outside of the loop
        output[0] = output_dct2[0];
        output[1] = output_dct4_even[0];
        output[2] = output_dct2[1];

        for i in 1..quarter_len {
            let dct4_cos_output = output_dct4_even[i];
            let dct4_sin_output = if (i + quarter_len) % 2 == 0 {
                -output_dct4_odd[quarter_len - i]
            } else {
                output_dct4_odd[quarter_len - i]
            };

            output[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
            output[i * 4] = output_dct2[i * 2];

            output[i * 4 + 1] =  dct4_cos_output - dct4_sin_output;
            output[i * 4 + 2] = output_dct2[i * 2 + 1];
        }

        output[len - 1] = -output_dct4_odd[0];
    }
}

impl<T: DCTnum> DCT2<T> for DCT2SplitRadix<T> {
    fn process(&mut self, input: &mut [T], output: &mut [T]) {
        assert!(input.len() == self.len());

        self.process_step(input, output, 1);
    }
}
impl<T> Length for DCT2SplitRadix<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 2
    }
}




#[cfg(test)]
mod test {
    use super::*;
    use dct2::DCT2Naive;

    use test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the DCT2 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct2_splitradix() {

        for i in 2..6 {
            let size = 1 << i;
            println!("len: {}", size);

            let mut expected_input = random_signal(size);
            let mut actual_input = expected_input.clone();

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT2Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let mut dct = DCT2SplitRadix::new(size);
            dct.process(&mut actual_input, &mut actual_output);

            println!("input:       {:?}", expected_input);
            println!("expected:    {:?}", expected_output);
            println!("fast output: {:?}", actual_output);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }
}
