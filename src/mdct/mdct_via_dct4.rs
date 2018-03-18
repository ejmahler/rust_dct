use rustfft::Length;

use dct4::DCT4;
use mdct::MDCT;
use DCTnum;

/// MDCT implementation that converts the problem to a DCT Type 4 of the same size.
///
/// It is much easier to express a MDCT as a DCT Type 4 than it is to express it as a FFT, so converting the MDCT
/// to a DCT4 before converting it to a FFT results in greatly simplified code
///
/// ~~~
/// // Computes a MDCT of input size 1234 via a DCT4, using the MP3 window function
/// use rustdct::mdct::{MDCT, MDCTViaDCT4, window_fn};
/// use rustdct::DCTplanner;
///
/// let input:  Vec<f32> = vec![0f32; 1234 * 2];
/// let mut output: Vec<f32> = vec![0f32; 1234];
///
/// let mut planner = DCTplanner::new();
/// let mut inner_dct4 = planner.plan_dct4(1234);
/// let mut dct = MDCTViaDCT4::new(inner_dct4, window_fn::mp3);
/// dct.process(&input, &mut output);
/// ~~~
pub struct MDCTViaDCT4<T> {
    dct: Box<DCT4<T>>,
    dct_buffer: Box<[T]>,
    window: Box<[T]>,
}

impl<T: DCTnum> MDCTViaDCT4<T> {
    /// Creates a new MDCT context that will process signals of length `inner_dct.len() * 2`, resulting in outputs of length `inner_dct.len()`
    ///
    /// `inner_dct.len()` must be even.
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    pub fn new<F>(inner_dct: Box<DCT4<T>>, window_fn: F) -> Self
    where
        F: FnOnce(usize) -> Vec<T>,
    {
        let len = inner_dct.len();

        assert!(len % 2 == 0, "The MDCT length must be even");

        let window = window_fn(len * 2);
        assert_eq!(
            window.len(),
            len * 2,
            "Window function returned incorrect number of values"
        );

        Self {
            dct: inner_dct,
            dct_buffer: vec![T::zero(); len].into_boxed_slice(),
            window: window.into_boxed_slice(),
        }
    }
}
impl<T: DCTnum> MDCT<T> for MDCTViaDCT4<T> {
    fn process_split(&mut self, input_a: &[T], input_b: &[T], output: &mut [T]) {
        assert_eq!(input_a.len(), self.len());

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
        for (element, (cr_val, d_val)) in
            self.dct_buffer.iter_mut().zip(
                group_c_rev_iter.zip(group_d_iter),
            )
        {
            *element = -cr_val - d_val;
        }

        //the second half of the dct input is is A - Br
        for (element, (a_val, br_val)) in
            self.dct_buffer[group_size..].iter_mut().zip(
                group_a_iter.zip(
                    group_b_rev_iter,
                ),
            )
        {
            *element = a_val - br_val;
        }

        self.dct.process(&mut self.dct_buffer, output);
    }
}
impl<T> Length for MDCTViaDCT4<T> {
    fn len(&self) -> usize {
        self.dct_buffer.len()
    }
}






#[cfg(test)]
mod unit_tests {
    use super::*;

    use dct4::DCT4Naive;
    use mdct::MDCTNaive;
    use mdct::window_fn;
    use test_utils::{compare_float_vectors, random_signal};

    /// Verify that our fast implementation of the MDCT and IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_mdct_via_dct4() {
        for current_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
            for i in 1..11 {
                let input_len = i * 4;
                let output_len = i * 2;

                let input = random_signal(input_len);

                let mut naive_output = vec![0f32; output_len];
                let mut fast_output = vec![0f32; output_len];


                let mut naive_mdct = MDCTNaive::new(output_len, current_window_fn);

                let inner_dct4 = Box::new(DCT4Naive::new(output_len));
                let mut fast_mdct = MDCTViaDCT4::new(inner_dct4, current_window_fn);

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
