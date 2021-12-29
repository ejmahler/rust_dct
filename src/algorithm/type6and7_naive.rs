use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::RequiredScratch;
use crate::{Dct6, Dct6And7, Dct7, DctNum, Dst6, Dst6And7, Dst7};

/// Naive O(n^2 ) DCT Type 6 and DCT Type 7 implementation
///
/// ~~~
/// // Computes a naive DCT6 and DCT7 of size 23
/// use rustdct::{Dct6, Dct7};
/// use rustdct::algorithm::Dct6And7Naive;
///
/// let len = 23;
/// let naive = Dct6And7Naive::new(len);
///
/// let mut dct6_buffer = vec![0f32; len];
/// naive.process_dct6(&mut dct6_buffer);
///
/// let mut dct7_buffer = vec![0f32; len];
/// naive.process_dct7(&mut dct7_buffer);
/// ~~~
pub struct Dct6And7Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dct6And7Naive<T> {
    /// Creates a new DCT6 and DCT7 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 - 1) as f64;

        let twiddles: Vec<T> = (0..len * 4 - 2)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dct6<T> for Dct6And7Naive<T> {
    fn process_dct6_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        scratch[scratch.len() - 1] = scratch[scratch.len() - 1] * T::half();
        buffer[0] = scratch.iter().fold(T::zero(), |acc, e| acc + *e);

        for k in 1..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2;
            let mut twiddle_index = k;

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
impl<T: DctNum> Dct7<T> for Dct6And7Naive<T> {
    fn process_dct7_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        scratch[0] = scratch[0] * T::half();

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = scratch[0];

            let twiddle_stride = k * 2 + 1;
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
impl<T> RequiredScratch for Dct6And7Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
impl<T: DctNum> Dct6And7<T> for Dct6And7Naive<T> {}
impl<T> Length for Dct6And7Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() + 2) / 4
    }
}

/// Naive O(n^2 ) DST Type 6 and DST Type 7 implementation
///
/// ~~~
/// // Computes a naive DST6 and DST7 of size 23
/// use rustdct::{Dst6, Dst7};
/// use rustdct::algorithm::Dst6And7Naive;
///
/// let len = 23;
/// let naive = Dst6And7Naive::new(len);
///
/// let mut dst6_buffer = vec![0f32; len];
/// naive.process_dst6(&mut dst6_buffer);
///
/// let mut dst7_buffer = vec![0f32; len];
/// naive.process_dst7(&mut dst7_buffer);
/// ~~~
pub struct Dst6And7Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: DctNum> Dst6And7Naive<T> {
    /// Creates a new DST6 and DST7 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 + 1) as f64;

        let twiddles: Vec<T> = (0..len * 4 + 2)
            .map(|i| (constant_factor * (i as f64)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
        }
    }
}

impl<T: DctNum> Dst6<T> for Dst6And7Naive<T> {
    fn process_dst6_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = (k + 1) * 2;
            let mut twiddle_index = k + 1;

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
impl<T: DctNum> Dst7<T> for Dst6And7Naive<T> {
    fn process_dst7_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());
        scratch.copy_from_slice(buffer);

        for k in 0..buffer.len() {
            let output_cell = buffer.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2 + 1;
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
impl<T> RequiredScratch for Dst6And7Naive<T> {
    fn get_scratch_len(&self) -> usize {
        self.len()
    }
}
impl<T: DctNum> Dst6And7<T> for Dst6And7Naive<T> {}
impl<T> Length for Dst6And7Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() - 2) / 4
    }
}
