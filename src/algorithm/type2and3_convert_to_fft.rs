use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftDirection, Length};

use crate::common::dct_error_inplace;
use crate::{array_utils::into_complex_mut, twiddles, RequiredScratch};
use crate::{Dct2, Dct3, DctNum, Dst2, Dst3, TransformType2And3};

/// DCT2, DST2, DCT3, and DST3 implementation that converts the problem into a FFT of the same size
///
/// ~~~
/// // Computes a O(NlogN) DCT2, DST2, DCT3, and DST3 of size 1234 by converting them to FFTs
/// use rustdct::{Dct2, Dst2, Dct3, Dst3};
/// use rustdct::algorithm::Type2And3ConvertToFft;
/// use rustdct::rustfft::FftPlanner;
///
/// let len = 1234;
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(len);
///
/// let dct = Type2And3ConvertToFft::new(fft);
///
/// let mut dct2_buffer = vec![0f32; len];
/// dct.process_dct2(&mut dct2_buffer);
///
/// let mut dst2_buffer = vec![0f32; len];
/// dct.process_dst2(&mut dst2_buffer);
///
/// let mut dct3_buffer = vec![0f32; len];
/// dct.process_dct3(&mut dct3_buffer);
///
/// let mut dst3_buffer = vec![0f32; len];
/// dct.process_dst3(&mut dst3_buffer);
/// ~~~
pub struct Type2And3ConvertToFft<T> {
    fft: Arc<dyn Fft<T>>,
    twiddles: Box<[Complex<T>]>,

    scratch_len: usize,
}

impl<T: DctNum> Type2And3ConvertToFft<T> {
    /// Creates a new DCT2, DST2, DCT3, and DST3 context that will process signals of length `inner_fft.len()`.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            inner_fft.fft_direction(),
            FftDirection::Forward,
            "The 'DCT type 2 via FFT' algorithm requires a forward FFT, but an inverse FFT was provided"
        );

        let len = inner_fft.len();

        let twiddles: Vec<Complex<T>> = (0..len)
            .map(|i| twiddles::single_twiddle(i, len * 4))
            .collect();

        let scratch_len = 2 * (len + inner_fft.get_inplace_scratch_len());

        Self {
            fft: inner_fft,
            twiddles: twiddles.into_boxed_slice(),
            scratch_len,
        }
    }
}

impl<T: DctNum> Dct2<T> for Type2And3ConvertToFft<T> {
    fn process_dct2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        // the first half of the array will be the even elements, in order
        let even_end = (buffer.len() + 1) / 2;
        for i in 0..even_end {
            fft_buffer[i] = Complex::from(buffer[i * 2]);
        }

        // the second half is the odd elements, in reverse order
        if self.len() > 1 {
            let odd_end = self.len() - 1 - self.len() % 2;
            for i in 0..self.len() / 2 {
                fft_buffer[even_end + i] = Complex::from(buffer[odd_end - 2 * i]);
            }
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // apply a correction factor to the result
        for ((fft_entry, correction_entry), spectrum_entry) in fft_buffer
            .iter()
            .zip(self.twiddles.iter())
            .zip(buffer.iter_mut())
        {
            *spectrum_entry = (fft_entry * correction_entry).re;
        }
    }
}
impl<T: DctNum> Dst2<T> for Type2And3ConvertToFft<T> {
    fn process_dst2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        // the first half of the array will be the even elements, in order
        let even_end = (buffer.len() + 1) / 2;
        for i in 0..even_end {
            fft_buffer[i] = Complex::from(buffer[i * 2]);
        }

        // the second half is the odd elements, in reverse order and negated
        if self.len() > 1 {
            let odd_end = self.len() - 1 - self.len() % 2;
            for i in 0..self.len() / 2 {
                fft_buffer[even_end + i] = Complex::from(-buffer[odd_end - 2 * i]);
            }
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // apply a correction factor to the result, and put it in reversed order in the output buffer
        for ((fft_entry, correction_entry), spectrum_entry) in fft_buffer
            .iter()
            .zip(self.twiddles.iter())
            .zip(buffer.iter_mut().rev())
        {
            *spectrum_entry = (fft_entry * correction_entry).re;
        }
    }
}
impl<T: DctNum> Dct3<T> for Type2And3ConvertToFft<T> {
    fn process_dct3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        // compute the FFT buffer based on the correction factors
        fft_buffer[0] = Complex::from(buffer[0] * T::half());

        for (i, (fft_input_element, twiddle)) in fft_buffer
            .iter_mut()
            .zip(self.twiddles.iter())
            .enumerate()
            .skip(1)
        {
            let c = Complex {
                re: buffer[i],
                im: buffer[buffer.len() - i],
            };
            *fft_input_element = c * twiddle * T::half();
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // copy the first half of the fft output into the even elements of the buffer
        let even_end = (buffer.len() + 1) / 2;
        for i in 0..even_end {
            buffer[i * 2] = fft_buffer[i].re;
        }

        // copy the second half of the fft buffer into the odd elements, reversed
        if self.len() > 1 {
            let odd_end = self.len() - 1 - self.len() % 2;
            for i in 0..self.len() / 2 {
                buffer[odd_end - 2 * i] = fft_buffer[i + even_end].re;
            }
        }
    }
}
impl<T: DctNum> Dst3<T> for Type2And3ConvertToFft<T> {
    fn process_dst3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        // compute the FFT buffer based on the correction factors
        fft_buffer[0] = Complex::from(buffer[buffer.len() - 1] * T::half());

        for (i, (fft_input_element, twiddle)) in fft_buffer
            .iter_mut()
            .zip(self.twiddles.iter())
            .enumerate()
            .skip(1)
        {
            let c = Complex {
                re: buffer[buffer.len() - i - 1],
                im: buffer[i - 1],
            };
            *fft_input_element = c * twiddle * T::half();
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // copy the first half of the fft output into the even elements of the output
        let even_end = (self.len() + 1) / 2;
        for i in 0..even_end {
            buffer[i * 2] = fft_buffer[i].re;
        }

        // copy the second half of the fft output into the odd elements, reversed
        if self.len() > 1 {
            let odd_end = self.len() - 1 - self.len() % 2;
            for i in 0..self.len() / 2 {
                buffer[odd_end - 2 * i] = -fft_buffer[i + even_end].re;
            }
        }
    }
}
impl<T: DctNum> TransformType2And3<T> for Type2And3ConvertToFft<T> {}
impl<T> Length for Type2And3ConvertToFft<T> {
    fn len(&self) -> usize {
        self.twiddles.len()
    }
}
impl<T: DctNum> RequiredScratch for Type2And3ConvertToFft<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::Type2And3Naive;

    use crate::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FftPlanner;

    /// Verify that our fast implementation of the DCT2 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dct2_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Type2And3Naive::new(size);
            naive_dct.process_dct2(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dct = Type2And3ConvertToFft::new(fft_planner.plan_fft_forward(size));
            dct.process_dct2(&mut actual_buffer);

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

    /// Verify that our fast implementation of the DST2 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dst2_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dst = Type2And3Naive::new(size);
            naive_dst.process_dst2(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dst = Type2And3ConvertToFft::new(fft_planner.plan_fft_forward(size));
            dst.process_dst2(&mut actual_buffer);

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

    /// Verify that our fast implementation of the DCT3 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dct3_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Type2And3Naive::new(size);
            naive_dct.process_dct3(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dct = Type2And3ConvertToFft::new(fft_planner.plan_fft_forward(size));
            dct.process_dct3(&mut actual_buffer);

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

    /// Verify that our fast implementation of the DST3 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dst3_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dst = Type2And3Naive::new(size);
            naive_dst.process_dst3(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dst = Type2And3ConvertToFft::new(fft_planner.plan_fft_forward(size));
            dst.process_dst3(&mut actual_buffer);

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
