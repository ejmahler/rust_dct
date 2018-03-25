use std::f64;

use rustfft::Length;

use dct2::DCT2;
use common;

/// Naive O(n^2 ) DCT Type 2 implementation
///
/// This implementation is primarily used to test other DCT2 algorithms. In rare cases, this algorithm may be faster
/// than `DCT2ViaFFT`.
///
/// ~~~
/// // Computes a naive DCT2 of size 123
/// use rustdct::dct2::{DCT2, DCT2Naive};
///
/// let len = 123;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let dct = DCT2Naive::new(len);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT2Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DCTnum> DCT2Naive<T> {
    /// Creates a new DCT2 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len * 4)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DCTnum> DCT2<T> for DCT2Naive<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2;
            let mut twiddle_index = k;

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
impl<T> Length for DCT2Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
