use std::f64;

use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::RequiredScratch;
use crate::{Dct5, DctNum, Dst5};

/// Naive O(n^2 ) DCT Type 5 implementation
///
/// This implementation is primarily used to test other DCT5 algorithms.
///
/// ~~~
/// // Computes a naive DCT5 of size 23
/// use rustdct::Dct5;
/// use rustdct::algorithm::Dct5Naive;
///
/// let len = 23;
/// let mut buffer = vec![0f32; len];
///
/// let dct = Dct5Naive::new(len);
/// dct.process_dct5(&mut buffer);
/// ~~~
pub struct Dct5Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dct5Naive<T> {
    pub fn new(len: usize) -> Self {
        let constant_factor = f64::consts::PI / (len as f64 - 0.5);

        let twiddles: Vec<T> = (0..len * 2 - 1)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct5<T> for Dct5Naive<T> {
    fn process_dct5_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        scratch[0] = scratch[0] * T::half();
        buffer[0] = scratch.iter().fold(T::zero(), |acc, e| acc + *e);

        for k in 1..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = scratch[0];

            let twiddle_stride = k;
            let mut twiddle_index = twiddle_stride;

            for i in 1..scratch.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + scratch[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> RequiredScratch for Dct5Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
impl<T> Length for Dct5Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() + 1) / 2
    }
}

/// Naive O(n^2 ) DST Type 5 implementation
///
/// This implementation is primarily used to test other DST5 algorithms.
///
/// ~~~
/// // Computes a naive DST5 of size 23
/// use rustdct::Dst5;
/// use rustdct::algorithm::Dst5Naive;
///
/// let len = 23;
/// let mut buffer = vec![0f32; len];
///
/// let dst = Dst5Naive::new(len);
/// dst.process_dst5(&mut buffer);
/// ~~~
pub struct Dst5Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dst5Naive<T> {
    /// Creates a new DST5 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = f64::consts::PI / (len as f64 + 0.5);

        let twiddles: Vec<T> = (0..len * 2 + 1)
            .map(|i| (constant_factor * (i as f64)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dst5<T> for Dst5Naive<T> {
    fn process_dst5_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k + 1;
            let mut twiddle_index = twiddle_stride;

            for i in 0..scratch.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + scratch[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> RequiredScratch for Dst5Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
impl<T> Length for Dst5Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() - 1) / 2
    }
}
