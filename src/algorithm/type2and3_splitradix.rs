use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::{twiddles, DctNum, RequiredScratch};
use crate::{Dct2, Dct3, Dst2, Dst3, TransformType2And3};

/// DCT2, DCT3, DST2, and DST3 implemention that recursively divides the problem in half.
///
/// The problem size must be 2^n, n > 1
///
/// ~~~
/// // Computes a DCT Type 2 of size 1024
/// use rustdct::algorithm::Type2And3SplitRadix;
/// use rustdct::Dct2;
/// use rustdct::DctPlanner;
///
/// let len = 1024;
///
/// let mut planner = DctPlanner::new();
/// let quarter_dct = planner.plan_dct2(len / 4);
/// let half_dct = planner.plan_dct2(len / 2);
///
/// let dct = Type2And3SplitRadix::new(half_dct, quarter_dct);
///
/// let mut buffer = vec![0f32; len];
/// dct.process_dct2(&mut buffer);
/// ~~~
pub struct Type2And3SplitRadix<T> {
    half_dct: Arc<dyn TransformType2And3<T>>,
    quarter_dct: Arc<dyn TransformType2And3<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: DctNum> Type2And3SplitRadix<T> {
    /// Creates a new DCT2, DCT3, DST2, and DST3 context that will process signals of length `half_dct.len() * 2`
    pub fn new(
        half_dct: Arc<dyn TransformType2And3<T>>,
        quarter_dct: Arc<dyn TransformType2And3<T>>,
    ) -> Self {
        let half_len = half_dct.len();
        let quarter_len = quarter_dct.len();
        let len = half_len * 2;

        assert!(
            len.is_power_of_two() && len > 2,
            "The DCT2SplitRadix algorithm requires a power-of-two input size greater than two. Got {}", len 
        );
        assert_eq!(half_len, quarter_len * 2,
            "half_dct.len() must be 2 * quarter_dct.len(). Got half_dct.len()={}, quarter_dct.len()={}", half_len, quarter_len
        );

        let twiddles: Vec<Complex<T>> = (0..(len / 4))
            .map(|i| twiddles::single_twiddle(2 * i + 1, len * 4).conj())
            .collect();

        Self {
            half_dct: half_dct,
            quarter_dct: quarter_dct,
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct2<T> for Type2And3SplitRadix<T> {
    fn process_dct2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();
        let half_len = len / 2;
        let quarter_len = len / 4;

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        let (input_dct2, input_dct4) = scratch.split_at_mut(half_len);
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

        for i in 0..quarter_len {
            
                let input_bottom = unsafe { *buffer.get_unchecked(i) };
                let input_top = unsafe { *buffer.get_unchecked(len - i - 1) };

                let input_half_bottom = unsafe { *buffer.get_unchecked(half_len - i - 1) };
                let input_half_top = unsafe { *buffer.get_unchecked(half_len + i) };

                //prepare the inner DCT2
                unsafe {*input_dct2.get_unchecked_mut(i)  = input_top + input_bottom };
                unsafe {*input_dct2.get_unchecked_mut(half_len - i - 1)  =
                    input_half_bottom + input_half_top };

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;
                let twiddle = unsafe { self.twiddles.get_unchecked(i) };

                let cos_input = lower_dct4 * twiddle.re + upper_dct4 * twiddle.im;
                let sin_input = upper_dct4 * twiddle.re - lower_dct4 * twiddle.im;

                unsafe {*input_dct4_even.get_unchecked_mut(i) = cos_input };
                unsafe {*input_dct4_odd.get_unchecked_mut(quarter_len - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input } };
            
        }

        // compute the recursive DCT2s, using the original buffer as scratch space
        self.half_dct.process_dct2_with_scratch(input_dct2, buffer);
        self.quarter_dct
            .process_dct2_with_scratch(input_dct4_even, buffer);
        self.quarter_dct
            .process_dct2_with_scratch(input_dct4_odd, buffer);

        unsafe {
            //post process the 3 DCT2 outputs. the first few and the last will be done outside of the loop
            *buffer.get_unchecked_mut(0) = *input_dct2.get_unchecked(0);
            *buffer.get_unchecked_mut(1) = *input_dct4_even.get_unchecked(0);
            *buffer.get_unchecked_mut(2) = *input_dct2.get_unchecked(1);

            for i in 1..quarter_len {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + quarter_len) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(quarter_len - i)
                } else {
                    *input_dct4_odd.get_unchecked(quarter_len - i)
                };

                *buffer.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                *buffer.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *buffer.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *buffer.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *buffer.get_unchecked_mut(len - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}
impl<T: DctNum> Dst2<T> for Type2And3SplitRadix<T> {
    fn process_dst2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        for i in 0..(self.len() / 2) {
            buffer[2 * i + 1] = buffer[2 * i + 1].neg();
        }

        self.process_dct2_with_scratch(buffer, scratch);

        buffer.reverse();
    }
}
impl<T: DctNum> Dct3<T> for Type2And3SplitRadix<T> {
    fn process_dct3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = buffer.len();
        let half_len = len / 2;
        let quarter_len = len / 4;

        // divide the output into 3 sub-lists to use for our inner DCTs, one of size N/2 and two of size N/4
        let (recursive_input_evens, recursive_input_odds) = scratch.split_at_mut(half_len);
        let (recursive_input_n1, recursive_input_n3) =
            recursive_input_odds.split_at_mut(quarter_len);

        // do the same pre-loop setup as DCT4ViaDCT3, and since we're skipping the first iteration of the loop we
        // to also set up the corresponding evens cells
        recursive_input_evens[0] = buffer[0];
        recursive_input_evens[1] = buffer[2];
        recursive_input_n1[0] = buffer[1] * T::two();
        recursive_input_n3[0] = buffer[len - 1] * T::two();

        // populate the recursive input arrays
        for i in 1..quarter_len {
            let k = 4 * i;

            unsafe {
                // the evens are the easy ones - just copy straight over
                *recursive_input_evens.get_unchecked_mut(i * 2) = *buffer.get_unchecked(k);
                *recursive_input_evens.get_unchecked_mut(i * 2 + 1) = *buffer.get_unchecked(k + 2);

                // for the odd ones we're going to do the same addition/subtraction we do in the setup for DCT4ViaDCT3
                *recursive_input_n1.get_unchecked_mut(i) =
                    *buffer.get_unchecked(k - 1) + *buffer.get_unchecked(k + 1);
                *recursive_input_n3.get_unchecked_mut(quarter_len - i) =
                    *buffer.get_unchecked(k - 1) - *buffer.get_unchecked(k + 1);
            }
        }

        //perform our recursive DCTs, using the original buffer as scratch space
        self.half_dct
            .process_dct3_with_scratch(recursive_input_evens, buffer);
        self.quarter_dct
            .process_dct3_with_scratch(recursive_input_n1, buffer);
        self.quarter_dct
            .process_dct3_with_scratch(recursive_input_n3, buffer);

        //merge the results. we're going to combine 2 separate things:
        // - merging the two smaller DCT3 outputs into a DCT4 output
        // - marging the DCT4 outputand the larger DCT3 output into the final output
        for i in 0..quarter_len {
            let twiddle = self.twiddles[i];
            let cosine_value = recursive_input_n1[i];

            // flip the sign of every other sine value to finish the job of using a DCT3 to compute a DST3
            let sine_value = if i % 2 == 0 {
                recursive_input_n3[i]
            } else {
                -recursive_input_n3[i]
            };

            let lower_dct4 = cosine_value * twiddle.re + sine_value * twiddle.im;
            let upper_dct4 = cosine_value * twiddle.im - sine_value * twiddle.re;

            unsafe {
                let lower_dct3 = *recursive_input_evens.get_unchecked(i);
                let upper_dct3 = *recursive_input_evens.get_unchecked(half_len - i - 1);

                *buffer.get_unchecked_mut(i) = lower_dct3 + lower_dct4;
                *buffer.get_unchecked_mut(len - i - 1) = lower_dct3 - lower_dct4;

                *buffer.get_unchecked_mut(half_len - i - 1) = upper_dct3 + upper_dct4;
                *buffer.get_unchecked_mut(half_len + i) = upper_dct3 - upper_dct4;
            }
        }
    }
}
impl<T: DctNum> Dst3<T> for Type2And3SplitRadix<T> {
    fn process_dst3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        buffer.reverse();

        self.process_dct3_with_scratch(buffer, scratch);

        for i in 0..(self.len() / 2) {
            buffer[2 * i + 1] = buffer[2 * i + 1].neg();
        }
    }
}
impl<T: DctNum> TransformType2And3<T> for Type2And3SplitRadix<T> {}
impl<T> Length for Type2And3SplitRadix<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 4
    }
}
impl<T> RequiredScratch for Type2And3SplitRadix<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::Type2And3Naive;

    use crate::test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the DCT2 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct2_splitradix() {
        for i in 2..8 {
            let size = 1 << i;
            println!("len: {}", size);

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Type2And3Naive::new(size);
            naive_dct.process_dct2(&mut expected_buffer);

            let quarter_dct = Arc::new(Type2And3Naive::new(size / 4));
            let half_dct = Arc::new(Type2And3Naive::new(size / 2));

            let dct = Type2And3SplitRadix::new(half_dct, quarter_dct);
            dct.process_dct2(&mut actual_buffer);

            println!("expected:    {:?}", expected_buffer);
            println!("fast output: {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }

    /// Verify that our fast implementation of the DCT3 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct3_splitradix() {
        for i in 2..8 {
            let size = 1 << i;
            println!("len: {}", size);

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Type2And3Naive::new(size);
            naive_dct.process_dct3(&mut expected_buffer);

            let quarter_dct = Arc::new(Type2And3Naive::new(size / 4));
            let half_dct = Arc::new(Type2And3Naive::new(size / 2));

            let dct = Type2And3SplitRadix::new(half_dct, quarter_dct);
            dct.process_dct3(&mut actual_buffer);

            println!("expected:    {:?}", expected_buffer);
            println!("fast output: {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }
}
