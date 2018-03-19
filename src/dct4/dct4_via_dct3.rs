use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::Length;

use twiddles;
use common;
use dct4::DCT4;
use dct3::DCT3;

/// DCT Type 4 implementation that converts the problem into two DCT type 3 of half size. If the inner DCT3 is 
/// O(nlogn), then so is this.
///
/// This algorithm can only be used if the problem size is even.
///
/// ~~~
/// // Computes a DCT Type 4 of size 1234
/// use std::sync::Arc;
/// use rustdct::dct4::{DCT4, DCT4ViaDCT3};
/// use rustdct::DCTplanner;
///
/// let len = 1234;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let mut planner = DCTplanner::new();
/// let inner_dct3 = planner.plan_dct3(len / 2);
/// 
/// let dct = DCT4ViaDCT3::new(inner_dct3);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT4ViaDCT3<T> {
    inner_dct: Arc<DCT3<T>>,
    twiddles: Box<[Complex<T>]>,
}

impl<T: common::DCTnum> DCT4ViaDCT3<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_dct.len() * 2`.
    pub fn new(inner_dct: Arc<DCT3<T>>) -> Self {
        let inner_len = inner_dct.len();
        let len = inner_len * 2;


        let twiddles: Vec<Complex<T>> = (0..inner_len)
            .map(|i| twiddles::single_twiddle(2 * i + 1, len * 8, true))
            .collect();

        Self {
            inner_dct: inner_dct,
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}
impl<T: common::DCTnum> DCT4<T> for DCT4ViaDCT3<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let len = self.len();
        let inner_len = len / 2;

        //pre-process the input by splitting into into two arrays, one for the inner DCT3, and the other for the DST3
        let (mut output_left, mut output_right) = output.split_at_mut(inner_len);

        output_left[0] = input[0] * T::from_usize(2).unwrap();
        output_right[0] = input[len - 1] * T::from_usize(2).unwrap();

        for k in 1..inner_len {
            output_left[k] =              input[2 * k - 1] + input[2 * k];
            output_right[inner_len - k] = input[2 * k - 1] - input[2 * k];
        }

        //run the two inner DCTs on our separated arrays
        let (mut inner_result_cosine, mut inner_result_sine) = input.split_at_mut(inner_len);

        self.inner_dct.process(&mut output_left, &mut inner_result_cosine);
        self.inner_dct.process(&mut output_right, &mut inner_result_sine);

        //post-process the data by combining it back into a single array
        for k in 0..inner_len {
            let twiddle = self.twiddles[k];

            let cosine_value = inner_result_cosine[k];

            // flip the sign of every other sine value to finish the job of using a DCT3 to compute a DST3
            let sine_value = if k % 2 == 0 {
                inner_result_sine[k]
            } else {
                -inner_result_sine[k]
            };

            output_left[k] =                  cosine_value * twiddle.re + sine_value * twiddle.im;
            output_right[inner_len - 1 - k] = cosine_value * twiddle.im - sine_value * twiddle.re;
        }
    }
}
impl<T> Length for DCT4ViaDCT3<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 2
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use test_utils::{compare_float_vectors, random_signal};
    use dct4::DCT4Naive;
    use dct3::DCT3Naive;

    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct4_via_dct3() {
        for inner_size in 1..20 {
            let size = inner_size * 2;

            let mut expected_input = random_signal(size);
            let mut actual_input = expected_input.clone();

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            
            let mut naive_dct4 = DCT4Naive::new(size);
            naive_dct4.process(&mut expected_input, &mut expected_output);

            let inner_dct3 =Arc::new(DCT3Naive::new(inner_size));
            let mut dct = DCT4ViaDCT3::new(inner_dct3);
            dct.process(&mut actual_input, &mut actual_output);

            let divided: Vec<f32> = expected_output
                .iter()
                .zip(actual_output.iter())
                .map(|(&a, b)| b / a)
                .collect();

            println!("");
            println!("expected: {:?}", expected_output);
            println!("actual:   {:?}", actual_output);
            println!("divided:  {:?}", divided);

            assert!(
                compare_float_vectors(&expected_output, &actual_output),
                "len = {}",
                size
            );
        }
    }
}
