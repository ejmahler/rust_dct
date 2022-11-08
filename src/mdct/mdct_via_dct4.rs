use std::sync::Arc;

use rustfft::Length;

use crate::common::mdct_error_inplace;
use crate::mdct::Mdct;
use crate::RequiredScratch;
use crate::{DctNum, TransformType4};

/// MDCT implementation that converts the problem to a DCT Type 4 of the same size.
///
/// It is much easier to express a MDCT as a DCT Type 4 than it is to express it as a FFT, so converting the MDCT
/// to a DCT4 before converting it to a FFT results in greatly simplified code
///
/// ~~~
/// // Computes a MDCT of input size 1234 via a DCT4, using the MP3 window function
/// use rustdct::mdct::{Mdct, MdctViaDct4, window_fn};
/// use rustdct::{DctPlanner, RequiredScratch};
///
/// let len = 1234;
///
/// let mut planner = DctPlanner::new();
/// let inner_dct4 = planner.plan_dct4(len);
///
/// let dct = MdctViaDct4::new(inner_dct4, window_fn::mp3);
///
/// let input = vec![0f32; len * 2];
/// let (input_a, input_b) = input.split_at(len);
/// let mut output = vec![0f32; len];
/// let mut scratch = vec![0f32; dct.get_scratch_len()];
///
/// dct.process_mdct_with_scratch(input_a, input_b, &mut output, &mut scratch);
/// ~~~
pub struct MdctViaDct4<T> {
    dct: Arc<dyn TransformType4<T>>,
    window: Box<[T]>,
    scratch_len: usize,
}

impl<T: DctNum> MdctViaDct4<T> {
    /// Creates a new MDCT context that will process signals of length `inner_dct.len() * 2`, with an output of length `inner_dct.len()`
    ///
    /// `inner_dct.len()` must be even.
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    pub fn new<F>(inner_dct: Arc<dyn TransformType4<T>>, window_fn: F) -> Self
    where
        F: FnOnce(usize) -> Vec<T>,
    {
        let len = inner_dct.len();

        assert!(len % 2 == 0, "The MDCT inner_dct.len() must be even");

        let window = window_fn(len * 2);
        assert_eq!(
            window.len(),
            len * 2,
            "Window function returned incorrect number of values"
        );

        Self {
            scratch_len: len + inner_dct.get_scratch_len(),
            dct: inner_dct,
            window: window.into_boxed_slice(),
        }
    }
}
impl<T: DctNum> Mdct<T> for MdctViaDct4<T> {
    fn process_mdct_with_scratch(
        &self,
        input_a: &[T],
        input_b: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) {
        let scratch = validate_buffers_mdct!(
            input_a,
            input_b,
            output,
            scratch,
            self.len(),
            self.get_scratch_len()
        );

        let group_size = self.len() / 2;

        //we're going to divide input_a into two subgroups, (a,b), and input_b into two subgroups: (c,d)
        //then scale them by the window function, then combine them into two subgroups: (-D-Cr, A-Br) where R means reversed
        let group_a_iter = input_a
            .iter()
            .zip(self.window.iter())
            .map(|(a, window_val)| *a * *window_val)
            .take(group_size);
        let group_b_rev_iter = input_a
            .iter()
            .zip(self.window.iter())
            .map(|(b, window_val)| *b * *window_val)
            .rev()
            .take(group_size);
        let group_c_rev_iter = input_b
            .iter()
            .zip(&self.window[self.len()..])
            .map(|(c, window_val)| *c * *window_val)
            .rev()
            .skip(group_size);
        let group_d_iter = input_b
            .iter()
            .zip(&self.window[self.len()..])
            .map(|(d, window_val)| *d * *window_val)
            .skip(group_size);

        //the first half of the dct input is -Cr - D
        for (element, (cr_val, d_val)) in output.iter_mut().zip(group_c_rev_iter.zip(group_d_iter))
        {
            *element = -cr_val - d_val;
        }

        //the second half of the dct input is is A - Br
        for (element, (a_val, br_val)) in output[group_size..]
            .iter_mut()
            .zip(group_a_iter.zip(group_b_rev_iter))
        {
            *element = a_val - br_val;
        }

        self.dct.process_dct4_with_scratch(output, scratch);
    }

    fn process_imdct_with_scratch(
        &self,
        input: &[T],
        output_a: &mut [T],
        output_b: &mut [T],
        scratch: &mut [T],
    ) {
        let scratch = validate_buffers_mdct!(
            input,
            output_a,
            output_b,
            scratch,
            self.len(),
            self.get_scratch_len()
        );

        let (dct_buffer, dct_scratch) = scratch.split_at_mut(self.len());
        dct_buffer.copy_from_slice(input);

        self.dct.process_dct4_with_scratch(dct_buffer, dct_scratch);

        let group_size = self.len() / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in output_a
            .iter_mut()
            .zip(&self.window[..])
            .zip(dct_buffer[group_size..].iter())
        {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in output_a
            .iter_mut()
            .zip(&self.window[..])
            .skip(group_size)
            .zip(dct_buffer[group_size..].iter().rev())
        {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in output_b
            .iter_mut()
            .zip(&self.window[self.len()..])
            .zip(dct_buffer[..group_size].iter().rev())
        {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in output_b
            .iter_mut()
            .zip(&self.window[self.len()..])
            .skip(group_size)
            .zip(dct_buffer[..group_size].iter())
        {
            *output = *output - *val * *window_val;
        }
    }
}
impl<T> Length for MdctViaDct4<T> {
    fn len(&self) -> usize {
        self.dct.len()
    }
}
impl<T> RequiredScratch for MdctViaDct4<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    use crate::algorithm::Type4Naive;
    use crate::mdct::window_fn;
    use crate::mdct::MdctNaive;
    use crate::test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the MDCT and IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_mdct_via_dct4() {
        for current_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
            for i in 1..11 {
                let input_len = i * 4;
                let output_len = i * 2;

                let input = random_signal(input_len);
                let (input_a, input_b) = input.split_at(output_len);

                let mut naive_output = vec![0f32; output_len];
                let mut fast_output = vec![0f32; output_len];

                let naive_mdct = MdctNaive::new(output_len, current_window_fn);

                let inner_dct4 = Arc::new(Type4Naive::new(output_len));
                let fast_mdct = MdctViaDct4::new(inner_dct4, current_window_fn);

                let mut naive_scratch = vec![0f32; naive_mdct.get_scratch_len()];
                let mut fast_scratch = vec![0f32; fast_mdct.get_scratch_len()];

                naive_mdct.process_mdct_with_scratch(
                    &input_a,
                    &input_b,
                    &mut naive_output,
                    &mut naive_scratch,
                );
                fast_mdct.process_mdct_with_scratch(
                    &input_a,
                    &input_b,
                    &mut fast_output,
                    &mut fast_scratch,
                );

                assert!(
                    compare_float_vectors(&naive_output, &fast_output),
                    "i = {}",
                    i
                );
            }
        }
    }

    /// Verify that our fast implementation of the MDCT and IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_imdct_via_dct4() {
        for current_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
            for i in 1..11 {
                let input_len = i * 2;
                let output_len = i * 4;

                let input = random_signal(input_len);

                // Fill both output buffers with ones, instead of zeroes, to verify that the IMDCT doesn't overwrite the output buffer
                let mut naive_output = vec![1f32; output_len];
                let (naive_output_a, naive_output_b) = naive_output.split_at_mut(input_len);

                let mut fast_output = vec![1f32; output_len];
                let (fast_output_a, fast_output_b) = fast_output.split_at_mut(input_len);

                let naive_mdct = MdctNaive::new(input_len, current_window_fn);

                let inner_dct4 = Arc::new(Type4Naive::new(input_len));
                let fast_mdct = MdctViaDct4::new(inner_dct4, current_window_fn);

                let mut naive_scratch = vec![0f32; naive_mdct.get_scratch_len()];
                let mut fast_scratch = vec![0f32; fast_mdct.get_scratch_len()];

                naive_mdct.process_imdct_with_scratch(
                    &input,
                    naive_output_a,
                    naive_output_b,
                    &mut naive_scratch,
                );
                fast_mdct.process_imdct_with_scratch(
                    &input,
                    fast_output_a,
                    fast_output_b,
                    &mut fast_scratch,
                );

                assert!(
                    compare_float_vectors(&naive_output, &fast_output),
                    "i = {}",
                    i
                );
            }
        }
    }
}
