use rustfft::num_complex::Complex;
use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::RequiredScratch;
use crate::{twiddles, DctNum};
use crate::{Dct4, Dst4, TransformType4};

/// Naive O(n^2 ) DCT Type 4 and DST Type 4 implementation
///
/// ~~~
/// // Computes a naive DCT4 of size 23
/// use rustdct::{Dct4, Dst4};
/// use rustdct::algorithm::Type4Naive;
///
/// let len = 23;
/// let naive = Type4Naive::new(len);
///
/// let mut dct4_buffer:  Vec<f32> = vec![0f32; len];
/// naive.process_dct4(&mut dct4_buffer);
///
/// let mut dst4_buffer:  Vec<f32> = vec![0f32; len];
/// naive.process_dst4(&mut dst4_buffer);
/// ~~~
pub struct Type4Naive<T> {
    twiddles: Box<[Complex<T>]>,
}

impl<T: DctNum> Type4Naive<T> {
    /// Creates a new DCT4 and DTS4 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let twiddles: Vec<Complex<T>> = (0..len * 4)
            .map(|i| twiddles::single_twiddle_halfoffset(i, len * 4))
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct4<T> for Type4Naive<T> {
    fn process_dct4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

            for i in 0..scratch.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + scratch[i] * twiddle.re;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T: DctNum> Dst4<T> for Type4Naive<T> {
    fn process_dst4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

            for i in 0..scratch.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell - scratch[i] * twiddle.im;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> RequiredScratch for Type4Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
impl<T: DctNum> TransformType4<T> for Type4Naive<T> {}
impl<T> Length for Type4Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
