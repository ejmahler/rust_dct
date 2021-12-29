use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftDirection, Length};

use crate::common::dct_error_inplace;
use crate::{array_utils::into_complex_mut, DctNum, RequiredScratch};
use crate::{Dst6, Dst6And7, Dst7};

/// DST6 and DST7 implementation that converts the problem into a FFT of the same size
///
/// ~~~
/// // Computes a O(NlogN) DST6 and DST7 of size 1234 by converting them to FFTs
/// use rustdct::{Dst6, Dst7};
/// use rustdct::algorithm::Dst6And7ConvertToFft;
/// use rustdct::rustfft::FftPlanner;
///
/// let len = 1234;
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(len * 2 + 1);
///
/// let dct = Dst6And7ConvertToFft::new(fft);
///
/// let mut dst6_buffer = vec![0f32; len];
/// dct.process_dst6(&mut dst6_buffer);
///
/// let mut dst7_buffer = vec![0f32; len];
/// dct.process_dst7(&mut dst6_buffer);
/// ~~~
pub struct Dst6And7ConvertToFft<T> {
    fft: Arc<dyn Fft<T>>,

    len: usize,
    scratch_len: usize,
    inner_fft_len: usize,
}

impl<T: DctNum> Dst6And7ConvertToFft<T> {
    /// Creates a new DST6 and DST7 context that will process signals of length `(inner_fft.len() - 1) / 2`.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();
        assert!(
            inner_fft_len % 2 == 1,
            "The 'DST6And7ConvertToFFT' algorithm requires an odd-len FFT. Provided len={}",
            inner_fft_len
        );
        assert_eq!(
            inner_fft.fft_direction(),
            FftDirection::Forward, "The 'DST6And7ConvertToFFT' algorithm requires a forward FFT, but an inverse FFT was provided");

        let len = (inner_fft_len - 1) / 2;

        Self {
            scratch_len: 2 * (inner_fft_len + inner_fft.get_inplace_scratch_len()),
            inner_fft_len,
            fft: inner_fft,
            len,
        }
    }
}
impl<T: DctNum> Dst6<T> for Dst6And7ConvertToFft<T> {
    fn process_dst6_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(self.inner_fft_len);

        // Copy the buffer to the odd imaginary components of the FFT inputs
        for i in 0..buffer.len() {
            fft_buffer[i * 2 + 1].im = buffer[i];
        }

        // inner fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // Copy the first half of the array to the odd-indexd elements
        let even_count = (buffer.len() + 1) / 2;
        let odd_count = buffer.len() - even_count;
        for i in 0..odd_count {
            let output_index = i * 2 + 1;
            buffer[output_index] = fft_buffer[i + 1].re;
        }

        // Copy the second half of the array to the reversed even-indexed elements
        for i in 0..even_count {
            let output_index = 2 * (even_count - i - 1);
            buffer[output_index] = fft_buffer[i + odd_count + 1].re;
        }
    }
}
impl<T: DctNum> Dst7<T> for Dst6And7ConvertToFft<T> {
    fn process_dst7_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(self.inner_fft_len);

        // Copy all the even-indexed elements to the back of the FFT buffer array
        let even_count = (buffer.len() + 1) / 2;
        for i in 0..even_count {
            let input_index = i * 2;
            let inner_index = buffer.len() + 1 + i;
            fft_buffer[inner_index] = Complex {
                re: buffer[input_index],
                im: T::zero(),
            };
        }
        // Copy all the odd-indexed elements in reverse order
        let odd_count = buffer.len() - even_count;
        for i in 0..odd_count {
            let input_index = 2 * (odd_count - i) - 1;
            let inner_index = buffer.len() + even_count + 1 + i;
            fft_buffer[inner_index] = Complex {
                re: buffer[input_index],
                im: T::zero(),
            };
        }
        // Copy the back of the array to the front, negated and reversed
        for i in 0..buffer.len() {
            fft_buffer[i + 1] = -fft_buffer[fft_buffer.len() - 1 - i];
        }

        // inner fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // copy buffer back
        for i in 0..buffer.len() {
            buffer[i] = fft_buffer[i * 2 + 1].im * T::half();
        }
    }
}
impl<T: DctNum> Dst6And7<T> for Dst6And7ConvertToFft<T> {}
impl<T: DctNum> RequiredScratch for Dst6And7ConvertToFft<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}
impl<T> Length for Dst6And7ConvertToFft<T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::Dst6And7Naive;

    use crate::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FftPlanner;

    /// Verify that our fast implementation of the DCT6 gives the same buffer as the naive version, for many different inputs
    #[test]
    fn test_dst6_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = random_signal(size);

            let naive_dst = Dst6And7Naive::new(size);
            naive_dst.process_dst6(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dst = Dst6And7ConvertToFft::new(fft_planner.plan_fft_forward(size * 2 + 1));
            assert_eq!(dst.len(), size);

            dst.process_dst6(&mut actual_buffer);

            println!("{}", size);
            println!("expected: {:?}", expected_buffer);
            println!("actual: {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }

    /// Verify that our fast implementation of the DST7 gives the same buffer as the naive version, for many different inputs
    #[test]
    fn test_dst7_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = random_signal(size);

            let naive_dst = Dst6And7Naive::new(size);
            naive_dst.process_dst7(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dst = Dst6And7ConvertToFft::new(fft_planner.plan_fft_forward(size * 2 + 1));
            assert_eq!(dst.len(), size);

            dst.process_dst7(&mut actual_buffer);

            println!("{}", size);
            println!("expected: {:?}", expected_buffer);
            println!("actual: {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }
}
