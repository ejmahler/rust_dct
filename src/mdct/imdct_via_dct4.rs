use rustfft::Length;

use dct4::DCT4;
use mdct::IMDCT;
use DCTnum;

pub struct IMDCTViaDCT4<T> {
    dct: Box<DCT4<T>>,
    dct_input: Box<[T]>,
    dct_output: Box<[T]>,
    window: Box<[T]>,
}

impl<T: DCTnum> IMDCTViaDCT4<T> {
    /// Creates a new MDCT context that will process signals of length `inner_dct.len() * 2`, resulting in outputs of length `inner_dct.len()`
    pub fn new<F>(inner_dct: Box<DCT4<T>>, window_fn: F) -> Self where F: Fn(usize) -> Vec<T> {
        let len = inner_dct.len();

        assert!(len % 2 == 0, "The IMDCT length must be even");

        Self {
            dct: inner_dct,
            dct_input: vec![T::zero(); len].into_boxed_slice(),
            dct_output: vec![T::zero(); len].into_boxed_slice(),
            window: window_fn(len * 2).into_boxed_slice(),
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
        for ((output, window_val), val) in output_a.iter_mut().zip(&self.window[..]).zip(self.dct_output[group_size..].iter()) {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in output_a.iter_mut().zip(&self.window[..]).skip(group_size).zip(self.dct_output[group_size..].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in output_b.iter_mut().zip(&self.window[self.len()..]).zip(self.dct_output[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in output_b.iter_mut().zip(&self.window[self.len()..]).skip(group_size).zip(self.dct_output[..group_size].iter()) {
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
    use ::test_utils::{compare_float_vectors, random_signal};

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

                let inner_dct4 = Box::new(DCT4Naive::new(input_len));
                let mut fast_mdct = IMDCTViaDCT4::new(inner_dct4, current_window_fn);

                naive_mdct.process(&input, &mut naive_output);
                fast_mdct.process(&input, &mut fast_output);

                assert!(compare_float_vectors(&naive_output, &fast_output), "i = {}", i);

            }
        }
    }
}