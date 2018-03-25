use std::f64;

use rustfft::Length;

use dct3::DCT3;
use common;

/// Naive O(n^2 ) DCT Type 3 implementation
///
/// This implementation is primarily used to test other DCT3 algorithms. In rare cases, this algorithm may be faster
/// than `DCT3ViaFFT`.
///
/// ~~~
/// // Computes a naive DCT3 of size 23
/// use rustdct::dct3::{DCT3, DCT3Naive};
///
/// let len = 23;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let dct = DCT3Naive::new(len);
/// dct.process(&mut input, &mut output);
/// ~~~
pub struct DCT3Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DCTnum> DCT3Naive<T> {
    /// Creates a new DCT3 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {

        let constant_factor = 0.5f64 * f64::consts::PI / (len as f64);

        let twiddles: Vec<T> = (0..len * 4)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DCTnum> DCT3<T> for DCT3Naive<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let half_first = T::from_f32(0.5f32).unwrap() * input[0];

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = half_first;

            let twiddle_stride = k * 2 + 1;
            let mut twiddle_index = twiddle_stride;

            for i in 1..input.len() {
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
impl<T> Length for DCT3Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
