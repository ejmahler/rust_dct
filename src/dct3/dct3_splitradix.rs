use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::Length;

use common;
use twiddles;
use dct3::DCT3;

/// DCT Type 3 implemention that recursively divides the problem in half. The problem size must be a power of two.
///
/// ~~~
/// // Computes a DCT Type 3 of size 1024
/// use std::sync::Arc;
/// use rustdct::dct3::{DCT3, DCT3SplitRadix};
/// use rustdct::DCTplanner;
///
/// let len = 1024;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let mut planner = DCTplanner::new();
/// let quarter_dct = planner.plan_dct3(len / 4);
/// let half_dct = planner.plan_dct3(len / 2);
/// 
/// let dct = DCT3SplitRadix::new(half_dct, quarter_dct);
/// dct.process_dct3(&mut input, &mut output);
/// ~~~
pub struct DCT3SplitRadix<T> {
    half_dct: Arc<DCT3<T>>,
    quarter_dct: Arc<DCT3<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: common::DCTnum> DCT3SplitRadix<T> {
    /// Creates a new DCT3 context that will process signals of length `len`.
    pub fn new(half_dct: Arc<DCT3<T>>, quarter_dct: Arc<DCT3<T>>) -> Self {
        let len = half_dct.len() * 2;
        assert!(
            len.is_power_of_two() && len > 2,
            "The DCT3SplitRadix algorithm requires a power-of-two input size greater than two. Got {}", len 
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
        
        {
            // divide the output into 3 sub-lists to use for our inner DCTs, one of size N/2 and two of size N/4
            let (recursive_input_evens, recursive_input_odds) = output.split_at_mut(half_len);
            let (recursive_input_n1, recursive_input_n3) = recursive_input_odds.split_at_mut(quarter_len);

            // do the same pre-loop setup as DCT4ViaDCT3, and since we're skipping the first iteration of the loop we
            // to also set up the corresponding evens cells
            recursive_input_evens[0] = input[0];
            recursive_input_evens[1] = input[2];
            recursive_input_n1[0] = input[1] * T::from_usize(2).unwrap();
            recursive_input_n3[0] = input[len - 1] * T::from_usize(2).unwrap();

            // populate the recursive input arrays
            for i in 1..quarter_len {
                let k = 4 * i;

                // the evens are the easy ones - just copy straight over
                *recursive_input_evens.get_unchecked_mut(i * 2) =     *input.get_unchecked(k);
                *recursive_input_evens.get_unchecked_mut(i * 2 + 1) = *input.get_unchecked(k + 2);

                // for the odd ones we're going to do the same addition/subtraction we do in the setup for DCT4ViaDCT3
                *recursive_input_n1.get_unchecked_mut(i) =               *input.get_unchecked(k - 1) + *input.get_unchecked(k + 1);
                *recursive_input_n3.get_unchecked_mut(quarter_len - i) = *input.get_unchecked(k - 1) - *input.get_unchecked(k + 1);
            }

            //now that we're done with the input, divide it up the same way we did the output
            let (recursive_output_evens, recursive_output_odds) = input.split_at_mut(half_len);
            let (recursive_output_n1, recursive_output_n3) = recursive_output_odds.split_at_mut(quarter_len);

            //perform our recursive DCTs
            self.half_dct.process_dct3(recursive_input_evens, recursive_output_evens);
            self.quarter_dct.process_dct3(recursive_input_n1, recursive_output_n1);
            self.quarter_dct.process_dct3(recursive_input_n3, recursive_output_n3);
        }

        //we want the input array to stay split, but to placate the borrow checker it's easier to just re-split it
        let (recursive_output_evens, recursive_output_odds) = input.split_at(half_len);
        let (recursive_output_n1, recursive_output_n3) = recursive_output_odds.split_at(quarter_len);

        //merge the results. we're going to combine 2 separate things:
        // - merging the two smaller DCT3 outputs into a DCT4 output
        // - marging the DCT4 outputand the larger DCT3 output into the final output
        for i in 0..quarter_len {
            let twiddle = self.twiddles.get_unchecked(i);
            let cosine_value = recursive_output_n1[i];

            // flip the sign of every other sine value to finish the job of using a DCT3 to compute a DST3
            let sine_value = if i % 2 == 0 {
                recursive_output_n3[i]
            } else {
                -recursive_output_n3[i]
            };

            let lower_dct4 = cosine_value * twiddle.re + sine_value * twiddle.im;
            let upper_dct4 = cosine_value * twiddle.im - sine_value * twiddle.re;

            let lower_dct3 = *recursive_output_evens.get_unchecked(i);
            let upper_dct3 = *recursive_output_evens.get_unchecked(half_len - i - 1);

            *output.get_unchecked_mut(i) =                lower_dct3 + lower_dct4;
            *output.get_unchecked_mut(len - i - 1) =      lower_dct3 - lower_dct4;

            *output.get_unchecked_mut(half_len - i - 1) = upper_dct3 + upper_dct4;
            *output.get_unchecked_mut(half_len + i) =     upper_dct3 - upper_dct4;
        }
    }
}

impl<T: common::DCTnum> DCT3<T> for DCT3SplitRadix<T> {
    fn process_dct3(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        unsafe {
            self.process_step(input, output);
        }
    }
}
impl<T> Length for DCT3SplitRadix<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 4
    }
}




#[cfg(test)]
mod test {
    use super::*;
    use dct3::DCT3Naive;

    use test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct3_splitradix() {
        for i in 2..6 {
            let size = 1 << i;
            let mut expected_input = random_signal(size);
            let mut actual_input = expected_input.clone();

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let mut naive_dct = DCT3Naive::new(size);
            naive_dct.process_dct3(&mut expected_input, &mut expected_output);

            let quarter_dct = Arc::new(DCT3Naive::new(size/4));
            let half_dct = Arc::new(DCT3Naive::new(size/2));

            let mut dct = DCT3SplitRadix::new(half_dct, quarter_dct);
            dct.process_dct3(&mut actual_input, &mut actual_output);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }
}
