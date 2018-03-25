use std::f64;

use rustfft::Length;

use dct4::DCT4;
use common;

/// Naive O(n^2 ) DCT Type 4 implementation
///
/// This implementation is primarily used to test other DCT4 algorithms. For small input sizes, this is actually
/// faster than `DCT4ViaFFT` because we don't have to pay the cost associated with converting the problem to a FFT.
///
/// ~~~
/// // Computes a naive DCT4 of size 23
/// use rustdct::dct4::{DCT4, DCT4Naive};
///
/// let len = 23;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let dct = DCT4Naive::new(len);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT4Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DCTnum> DCT4Naive<T> {
    /// Creates a new DCT4 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len * 4)
            .map(|i| (constant_factor * (i as f64 + 0.5_f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DCTnum> DCT4<T> for DCT4Naive<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

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
impl<T> Length for DCT4Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
