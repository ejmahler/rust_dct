use std::f64;

use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::RequiredScratch;
use crate::{Dct1, DctNum, Dst1};

/// Naive O(n^2 ) DCT Type 1 implementation
///
/// This implementation is primarily used to test other DCT1 algorithms. For small scratch sizes, this is actually
/// faster than `DCT1ViaFFT` because we don't have to pay the cost associated with converting the problem to a FFT.
///
/// ~~~
/// // Computes a naive DCT1 of size 23
/// use rustdct::Dct1;
/// use rustdct::algorithm::Dct1Naive;
///
/// let len = 23;
///
/// let dct = Dct1Naive::new(len);
///
/// let mut buffer = vec![0f32; len];
/// dct.process_dct1(&mut buffer);
/// ~~~
pub struct Dct1Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dct1Naive<T> {
    pub fn new(len: usize) -> Self {
        assert_ne!(len, 1, "DCT Type 1 is undefined for len == 1");

        let constant_factor = f64::consts::PI / ((len - 1) as f64);

        let twiddles: Vec<T> = (0..(len - 1) * 2)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct1<T> for Dct1Naive<T> {
    fn process_dct1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        let half = T::half();
        scratch[0] = scratch[0] * half;
        scratch[self.len() - 1] = scratch[self.len() - 1] * half;

        for k in 0..buffer.len() {
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
impl<T> Length for Dct1Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 2 + 1
    }
}
impl<T> RequiredScratch for Dct1Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}

/// Naive O(n^2 ) DST Type 1 implementation
///
/// This implementation is primarily used to test other DST1 algorithms.
///
/// ~~~
/// // Computes a naive DST1 of size 23
/// use rustdct::Dst1;
/// use rustdct::algorithm::Dst1Naive;
///
/// let len = 23;
///
/// let dst = Dst1Naive::new(len);
///
/// let mut buffer = vec![0f32; len];
/// dst.process_dst1(&mut buffer);
/// ~~~
pub struct Dst1Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dst1Naive<T> {
    /// Creates a new DST1 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = f64::consts::PI / ((len + 1) as f64);

        let twiddles: Vec<T> = (0..(len + 1) * 2)
            .map(|i| (constant_factor * (i as f64)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dst1<T> for Dst1Naive<T> {
    fn process_dst1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
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
impl<T> Length for Dst1Naive<T> {
    fn len(&self) -> usize {
        self.twiddles.len() / 2 - 1
    }
}
impl<T> RequiredScratch for Dst1Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
