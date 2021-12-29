use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftDirection;
use rustfft::{Fft, Length};

use crate::common::dct_error_inplace;
use crate::{array_utils::into_complex_mut, DctNum, RequiredScratch};
use crate::{Dct1, Dst1};

/// DCT Type 1 implementation that converts the problem into a FFT of size 2 * (n - 1)
///
/// ~~~
/// // Computes a DCT Type 1 of size 1234
/// use rustdct::Dct1;
/// use rustdct::algorithm::Dct1ConvertToFft;
/// use rustdct::rustfft::FftPlanner;
///
/// let len = 1234;
///
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(2 * (len - 1));
///
/// let dct = Dct1ConvertToFft::new(fft);
///
/// let mut buffer = vec![0f32; len];
/// dct.process_dct1(&mut buffer);
pub struct Dct1ConvertToFft<T> {
    fft: Arc<dyn Fft<T>>,

    len: usize,
    scratch_len: usize,
    inner_fft_len: usize,
}

impl<T: DctNum> Dct1ConvertToFft<T> {
    /// Creates a new DCT1 context that will process signals of length `inner_fft.len() / 2 + 1`.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();

        assert!(
            inner_fft_len % 2 == 0,
            "For DCT1 via FFT, the inner FFT size must be even. Got {}",
            inner_fft_len
        );
        assert_eq!(
            inner_fft.fft_direction(),
            FftDirection::Forward,
            "The 'DCT type 1 via FFT' algorithm requires a forward FFT, but an inverse FFT \
                 was provided"
        );

        let len = inner_fft_len / 2 + 1;

        Self {
            scratch_len: 2 * (inner_fft_len + inner_fft.get_inplace_scratch_len()),
            inner_fft_len,
            fft: inner_fft,
            len,
        }
    }
}

impl<T: DctNum> Dct1<T> for Dct1ConvertToFft<T> {
    fn process_dct1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(self.inner_fft_len);

        for (&input_val, fft_cell) in buffer.iter().zip(&mut fft_buffer[..buffer.len()]) {
            *fft_cell = Complex {
                re: input_val,
                im: T::zero(),
            };
        }
        for (&input_val, fft_cell) in buffer
            .iter()
            .rev()
            .skip(1)
            .zip(&mut fft_buffer[buffer.len()..])
        {
            *fft_cell = Complex {
                re: input_val,
                im: T::zero(),
            };
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // apply a correction factor to the result
        let half = T::half();
        for (fft_entry, output_val) in fft_buffer.iter().zip(buffer.iter_mut()) {
            *output_val = fft_entry.re * half;
        }
    }
}
impl<T: DctNum> RequiredScratch for Dct1ConvertToFft<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}
impl<T> Length for Dct1ConvertToFft<T> {
    fn len(&self) -> usize {
        self.len
    }
}

/// DST Type 1 implementation that converts the problem into a FFT of size 2 * (n + 1)
///
/// ~~~
/// // Computes a DST Type 1 of size 1234
/// use rustdct::Dst1;
/// use rustdct::algorithm::Dst1ConvertToFft;
/// use rustdct::rustfft::FftPlanner;
///
/// let len = 1234;
///
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(2 * (len + 1));
///
/// let dct = Dst1ConvertToFft::new(fft);
///
/// let mut buffer = vec![0f32; len];
/// dct.process_dst1(&mut buffer);
/// ~~~
pub struct Dst1ConvertToFft<T> {
    fft: Arc<dyn Fft<T>>,

    len: usize,
    scratch_len: usize,
    inner_fft_len: usize,
}

impl<T: DctNum> Dst1ConvertToFft<T> {
    /// Creates a new DST1 context that will process signals of length `inner_fft.len() / 2 - 1`.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        let inner_fft_len = inner_fft.len();

        assert!(
            inner_fft_len % 2 == 0,
            "For DCT1 via FFT, the inner FFT size must be even. Got {}",
            inner_fft_len
        );
        assert_eq!(
            inner_fft.fft_direction(),
            FftDirection::Forward,
            "The 'DCT type 1 via FFT' algorithm requires a forward FFT, but an inverse FFT \
                 was provided"
        );

        let len = inner_fft_len / 2 - 1;

        Self {
            scratch_len: 2 * (inner_fft_len + inner_fft.get_inplace_scratch_len()),
            inner_fft_len,
            fft: inner_fft,
            len,
        }
    }
}

impl<T: DctNum> Dst1<T> for Dst1ConvertToFft<T> {
    fn process_dst1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(self.inner_fft_len);

        // the first half of the FFT input will be a 0, followed by the input array
        for (input_val, fft_cell) in buffer.iter().zip(fft_buffer.iter_mut().skip(1)) {
            *fft_cell = Complex::from(input_val);
        }

        // the second half of the FFT input will be a 0, followed by the input array, reversed and negated
        for (input_val, fft_cell) in buffer.iter().zip(fft_buffer.iter_mut().rev()) {
            *fft_cell = Complex::from(-*input_val);
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        // apply a correction factor to the result
        let half = T::half();
        for (fft_entry, output_val) in fft_buffer.iter().rev().zip(buffer.iter_mut()) {
            *output_val = fft_entry.im * half;
        }
    }
}
impl<T: DctNum> RequiredScratch for Dst1ConvertToFft<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}
impl<T> Length for Dst1ConvertToFft<T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::{Dct1Naive, Dst1Naive};

    use crate::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FftPlanner;

    /// Verify that our fast implementation of the DCT1 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct1_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Dct1Naive::new(size);
            naive_dct.process_dct1(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let inner_fft = fft_planner.plan_fft_forward((size - 1) * 2);
            println!("size: {}", size);
            println!("inner fft len: {}", inner_fft.len());

            let dct = Dct1ConvertToFft::new(inner_fft);
            println!("dct len: {}", dct.len());
            dct.process_dct1(&mut actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }

    /// Verify that our fast implementation of the DST1 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dst1_via_fft() {
        for size in 2..20 {
            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct = Dst1Naive::new(size);
            naive_dct.process_dst1(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let inner_fft = fft_planner.plan_fft_forward((size + 1) * 2);
            println!("size: {}", size);
            println!("inner fft len: {}", inner_fft.len());

            let dct = Dst1ConvertToFft::new(inner_fft);
            println!("dst len: {}", dct.len());
            dct.process_dst1(&mut actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }
}
