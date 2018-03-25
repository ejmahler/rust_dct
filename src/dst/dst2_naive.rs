use std::f64;

use rustfft::Length;

use dst::DST2;
use common;

/// Naive O(n^2 ) DST Type 2 implementation
///
/// This implementation is primarily used to test other DST2 algorithms.
///
/// ~~~
/// // Computes a naive DST2 of size 123
/// use rustdct::dst::{DST2, DST2Naive};
///
/// let len = 123;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let dst = DST2Naive::new(len);
/// dst.process(&mut input, &mut output);
/// ~~~
pub struct DST2Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DCTnum> DST2Naive<T> {
    /// Creates a new DST2 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len * 4)
            .map(|i| (constant_factor * (i as f64)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DCTnum> DST2<T> for DST2Naive<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = (k + 1) * 2;
            let mut twiddle_index = k + 1;

            for i in 0..input.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + input[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> Length for DST2Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
