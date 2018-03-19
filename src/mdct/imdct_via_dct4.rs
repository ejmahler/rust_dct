use std::sync::Arc;

use rustfft::Length;

use dct4::DCT4;
use mdct::IMDCT;
use DCTnum;

/// IMDCT implementation that converts the problem to a DCT Type 4 of the same size.
///
/// It is much easier to express a IMDCT as a DCT Type 4 than it is to express it as a FFT, so converting the IMDCT
/// to a DCT4 before converting it to a FFT results in greatly simplified code
///
/// ~~~
/// // Computes a IMDCT of input size 1234 via a DCT4, using the MP3 window function
/// use rustdct::mdct::{IMDCT, IMDCTViaDCT4, window_fn};
/// use rustdct::DCTplanner;
///
/// let input:  Vec<f32> = vec![0f32; 1234];
/// let mut output: Vec<f32> = vec![0f32; 1234 * 2];
///
/// let mut planner = DCTplanner::new();
/// let mut inner_dct4 = planner.plan_dct4(1234);
/// let mut dct = IMDCTViaDCT4::new(inner_dct4, window_fn::mp3);
/// dct.process(&input, &mut output);
/// ~~~
pub struct IMDCTViaDCT4<T> {
    dct: Arc<DCT4<T>>,
    dct_input: Box<[T]>,
    dct_output: Box<[T]>,
    window: Box<[T]>,
}

impl<T: DCTnum> IMDCTViaDCT4<T> {
    /// Creates a new IMDCT context that will process signals of input length `inner_dct.len()`, resulting in outputs of length `inner_dct.len() * 2`
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    pub fn new<F>(inner_dct: Arc<DCT4<T>>, window_fn: F) -> Self
    where
        F: FnOnce(usize) -> Vec<T>,
    {
        let len = inner_dct.len();

        assert!(
            len % 2 == 0,
            "The IMDCT inner_dct.len() must be even. Got {}",
            len
        );

        let window = window_fn(len * 2);
        assert_eq!(
            window.len(),
            len * 2,
            "Window function returned incorrect number of values"
        );

        Self {
            dct: inner_dct,
            dct_input: vec![T::zero(); len].into_boxed_slice(),
            dct_output: vec![T::zero(); len].into_boxed_slice(),
            window: window.into_boxed_slice(),
        }
    }
}
impl<T: DCTnum> IMDCT<T> for IMDCTViaDCT4<T> {
    fn process_split(&mut self, input: &[T], output_a: &mut [T], output_b: &mut [T]) {
        assert_eq!(input.len(), self.len());

        self.dct_input.copy_from_slice(input);

        self.dct.process(&mut self.dct_input, &mut self.dct_output);

        let group_size = self.len() / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in
            output_a.iter_mut().zip(&self.window[..]).zip(
                self.dct_output
                    [group_size..]
                    .iter(),
            )
        {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in
            output_a
                .iter_mut()
                .zip(&self.window[..])
                .skip(group_size)
                .zip(self.dct_output[group_size..].iter().rev())
        {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in
            output_b.iter_mut().zip(&self.window[self.len()..]).zip(
                self.dct_output[..group_size].iter().rev(),
            )
        {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in
            output_b
                .iter_mut()
                .zip(&self.window[self.len()..])
                .skip(group_size)
                .zip(self.dct_output[..group_size].iter())
        {
            *output = *output - *val * *window_val;
        }
    }
}
impl<T> Length for IMDCTViaDCT4<T> {
    fn len(&self) -> usize {
        self.dct_input.len()
    }
}






#[cfg(test)]
mod unit_tests {
    use super::*;

    use dct4::DCT4Naive;
    use mdct::IMDCTNaive;
    use mdct::window_fn;
    use test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the MDCT and IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_imdct_via_dct4() {
        for current_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
            for i in 1..11 {
                let input_len = i * 2;
                let output_len = i * 4;

                let input = random_signal(input_len);

                let mut naive_output = vec![0f32; output_len];
                let mut fast_output = vec![0f32; output_len];


                let mut naive_mdct = IMDCTNaive::new(input_len, current_window_fn);

                let inner_dct4 = Arc::new(DCT4Naive::new(input_len));
                let mut fast_mdct = IMDCTViaDCT4::new(inner_dct4, current_window_fn);

                naive_mdct.process(&input, &mut naive_output);
                fast_mdct.process(&input, &mut fast_output);

                assert!(
                    compare_float_vectors(&naive_output, &fast_output),
                    "i = {}",
                    i
                );

            }
        }
    }
}
