use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::Length;

use DCTnum;
use twiddles;
use dct2::DCT2;

/// DCT Type 2 implemention that recursively divides the problem in half. The problem size must be a power of two.
///
/// ~~~
/// // Computes a DCT Type 2 of size 1024
/// use std::sync::Arc;
/// use rustdct::dct2::{DCT2, DCT2Naive, DCT2SplitRadix};
///
/// let mut input:  Vec<f32> = vec![0f32; 1024];
/// let mut output: Vec<f32> = vec![0f32; 1024];
///
/// let quarter_dct = Arc::new(DCT2Naive::new(256));
/// let half_dct = Arc::new(DCT2Naive::new(512));
///
/// let mut dct = DCT2SplitRadix::new(half_dct, quarter_dct);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT2SplitRadix<T> {
    half_dct: Arc<DCT2<T>>,
    quarter_dct: Arc<DCT2<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: DCTnum> DCT2SplitRadix<T> {
    /// Creates a new DCT2 context that will process signals of length `len`
    pub fn new(half_dct: Arc<DCT2<T>>, quarter_dct: Arc<DCT2<T>>) -> Self {
        let len = half_dct.len() * 2;
        assert!(
            len.is_power_of_two() && len > 2,
            "The DCT2SplitRadix algorithm requires a power-of-two input size greater than two. Got {}", len 
        );

        let twiddles: Vec<Complex<T>> = (0..(len/4))
            .map(|i| twiddles::single_twiddle(2 * i + 1, len * 4, true))
            .collect();

        Self {
            half_dct: half_dct,
            quarter_dct: quarter_dct,
            twiddles: twiddles.into_boxed_slice(),
        }
    }

    // UNSAFE: Assumes that
    // - input.len() and output.len() are equal,
    // - input.len() and output.len() are equal to self.len()
    unsafe fn process_step(&self, input: &mut [T], output: &mut [T]) {
        let len = input.len();
        let half_len = len / 2;
        let quarter_len = len / 4;

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        {
            let (input_dct2, input_dct4) = output.split_at_mut(half_len);
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

            for i in 0..quarter_len {
                let input_bottom = *input.get_unchecked(i);
                let input_top =    *input.get_unchecked(len - i - 1);

                let input_half_bottom = *input.get_unchecked(half_len - i - 1);
                let input_half_top =    *input.get_unchecked(half_len + i);

                //prepare the inner DCT2
                *input_dct2.get_unchecked_mut(i) =                input_top + input_bottom;
                *input_dct2.get_unchecked_mut(half_len - i - 1) = input_half_bottom + input_half_top;

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;
                let twiddle = self.twiddles.get_unchecked(i);

                let cos_input = lower_dct4 * twiddle.re + upper_dct4 * twiddle.im;
                let sin_input = upper_dct4 * twiddle.re - lower_dct4 * twiddle.im;

                *input_dct4_even.get_unchecked_mut(i) = cos_input;
                *input_dct4_odd.get_unchecked_mut(quarter_len - i - 1) = if i % 2 == 0 {
                    sin_input
                } else {
                    -sin_input
                };
            }

            // compute the recursive DCT2s
            let (output_dct2, output_dct4) = input.split_at_mut(half_len);
            let (output_dct4_even, output_dct4_odd) = output_dct4.split_at_mut(quarter_len);

            self.half_dct.process(input_dct2, output_dct2);
            self.quarter_dct.process(input_dct4_even, output_dct4_even);
            self.quarter_dct.process(input_dct4_odd, output_dct4_odd);
        }



        let (output_dct2, output_dct4) = input.split_at(half_len);
        let (output_dct4_even, output_dct4_odd) = output_dct4.split_at(quarter_len);


        //post process the 3 DCT3 outputs/ the first few and the last will be done outside of the loop
        *output.get_unchecked_mut(0) = *output_dct2.get_unchecked(0);
        *output.get_unchecked_mut(1) = *output_dct4_even.get_unchecked(0);
        *output.get_unchecked_mut(2) = *output_dct2.get_unchecked(1);

        for i in 1..quarter_len {
            let dct4_cos_output = *output_dct4_even.get_unchecked(i);
            let dct4_sin_output = if (i + quarter_len) % 2 == 0 {
                -*output_dct4_odd.get_unchecked(quarter_len - i)
            } else {
                *output_dct4_odd.get_unchecked(quarter_len - i)
            };

            *output.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
            *output.get_unchecked_mut(i * 4) = *output_dct2.get_unchecked(i * 2);

            *output.get_unchecked_mut(i * 4 + 1) =  dct4_cos_output - dct4_sin_output;
            *output.get_unchecked_mut(i * 4 + 2) = *output_dct2.get_unchecked(i * 2 + 1);
        }

        *output.get_unchecked_mut(len - 1) = -*output_dct4_odd.get_unchecked(0);
    }
}

impl<T: DCTnum> DCT2<T> for DCT2SplitRadix<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        assert!(input.len() == self.len());

        unsafe {
            self.process_step(input, output);
        }
    }
}
impl<T> Length for DCT2SplitRadix<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 4
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

        for i in 2..8 {
            let size = 1 << i;
            println!("len: {}", size);

            let mut expected_input = random_signal(size);
            let mut actual_input = expected_input.clone();

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let naive_dct = DCT2Naive::new(size);
            naive_dct.process(&mut expected_input, &mut expected_output);

            let quarter_dct = Arc::new(DCT2Naive::new(size/4));
            let half_dct = Arc::new(DCT2Naive::new(size/2));

            let dct = DCT2SplitRadix::new(half_dct, quarter_dct);
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
