use std::f64;

use rustfft::Length;

use ::DCT5;
use common;


/// Naive O(n^2 ) DCT Type 5 implementation
///
/// This implementation is primarily used to test other DCT5 algorithms.
///
/// ~~~
/// // Computes a naive DCT5 of size 23
/// use rustdct::DCT5;
/// use rustdct::algorithm::DCT5Naive;
///
/// let len = 23;
/// let mut input:  Vec<f32> = vec![0f32; len];
/// let mut output: Vec<f32> = vec![0f32; len];
///
/// let dct = DCT5Naive::new(len);
/// dct.process_dct1(&mut input, &mut output);
/// ~~~
pub struct DCT5Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DCTnum> DCT5Naive<T> {
    pub fn new(len: usize) -> Self {
        let constant_factor = f64::consts::PI / (len as f64 - 0.5);

        let twiddles: Vec<T> = (0..len*2 - 1)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        DCT5Naive { twiddles: dbg!(twiddles.into_boxed_slice()) }
    }
}

impl<T: common::DCTnum> DCT5<T> for DCT5Naive<T> {
    fn process_dct5(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        input[0] = input[0] * T::half();
        output[0] = input.iter().fold(T::zero(), |acc, e| acc + *e);

        for k in 1..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = input[0];

            let twiddle_stride = k;
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
impl<T> Length for DCT5Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() + 1) / 2
    }
}
