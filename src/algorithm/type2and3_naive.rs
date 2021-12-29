use rustfft::num_complex::Complex;
use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::RequiredScratch;
use crate::{twiddles, DctNum};
use crate::{Dct2, Dct3, Dst2, Dst3, TransformType2And3};

/// Naive O(n^2 ) DCT Type 2, DST Type 2, DCT Type 3, and DST Type 3 implementation
///
/// ~~~
/// // Computes a naive DCT2, DST2, DCT3, and DST3 of size 23
/// use rustdct::{Dct2, Dst2, Dct3, Dst3};
/// use rustdct::algorithm::Type2And3Naive;
///
/// let len = 23;
/// let naive = Type2And3Naive::new(len);
///
/// let mut dct2_buffer = vec![0f32; len];
/// naive.process_dct2(&mut dct2_buffer);
///
/// let mut dst2_buffer = vec![0f32; len];
/// naive.process_dst2(&mut dst2_buffer);
///
/// let mut dct3_buffer = vec![0f32; len];
/// naive.process_dct3(&mut dct3_buffer);
///
/// let mut dst3_buffer = vec![0f32; len];
/// naive.process_dst3(&mut dst3_buffer);
/// ~~~
pub struct Type2And3Naive<T> {
    twiddles: Box<[Complex<T>]>,
}

impl<T: DctNum> Type2And3Naive<T> {
    /// Creates a new DCT2, DCT3, DST2, and DST3 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let twiddles: Vec<Complex<T>> = (0..len * 4)
            .map(|i| twiddles::single_twiddle(i, len * 4))
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct2<T> for Type2And3Naive<T> {
    fn process_dct2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2;
            let mut twiddle_index = k;

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
impl<T: DctNum> Dst2<T> for Type2And3Naive<T> {
    fn process_dst2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = (k + 1) * 2;
            let mut twiddle_index = k + 1;

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
impl<T: DctNum> Dct3<T> for Type2And3Naive<T> {
    fn process_dct3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        let half_first = T::half() * scratch[0];

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = half_first;

            let twiddle_stride = k * 2 + 1;
            let mut twiddle_index = twiddle_stride;

            for i in 1..scratch.len() {
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
impl<T: DctNum> Dst3<T> for Type2And3Naive<T> {
    fn process_dst3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        // scale the last scratch value by half before going into the loop
        scratch[scratch.len() - 1] = scratch[scratch.len() - 1] * T::half();

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2 + 1;
            let mut twiddle_index = twiddle_stride;

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
impl<T: DctNum> TransformType2And3<T> for Type2And3Naive<T> {}
impl<T> Length for Type2And3Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 4
    }
}
impl<T> RequiredScratch for Type2And3Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
