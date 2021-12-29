use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftDirection;
use rustfft::{Fft, Length};

use crate::common::dct_error_inplace;
use crate::{array_utils::into_complex_mut, DctNum, RequiredScratch};
use crate::{Dct4, Dst4, TransformType4};

/// DCT Type 4 and DST Type 4 implementation that converts the problem into a FFT of the same size.
///
/// This algorithm can only be used if the problem size is odd.
///
/// ~~~
/// // Computes a DCT Type 4 and DST Type 4 of size 1233
/// use rustdct::{Dct4, Dst4};
/// use rustdct::algorithm::Type4ConvertToFftOdd;
/// use rustdct::rustfft::FftPlanner;
///
/// let len = 1233;
///
/// let mut planner = FftPlanner::new();
/// let fft = planner.plan_fft_forward(len);
/// let dct = Type4ConvertToFftOdd::new(fft);
///
/// let mut dct4_buffer = vec![0f32; len];
/// dct.process_dct4(&mut dct4_buffer);
///
/// let mut dst4_buffer = vec![0f32; len];
/// dct.process_dst4(&mut dst4_buffer);
/// ~~~
pub struct Type4ConvertToFftOdd<T> {
    fft: Arc<dyn Fft<T>>,

    len: usize,
    scratch_len: usize,
}

impl<T: DctNum> Type4ConvertToFftOdd<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len()`. `inner_fft.len()` must be odd.
    pub fn new(inner_fft: Arc<dyn Fft<T>>) -> Self {
        assert_eq!(
            inner_fft.fft_direction(),
            FftDirection::Forward,
            "Type4ConvertToFFTOdd requires a forward FFT, but an inverse FFT was provided"
        );

        let len = inner_fft.len();

        assert!(
            len % 2 == 1,
            "Type4ConvertToFFTOdd size must be odd. Got {}",
            len
        );

        Self {
            scratch_len: 2 * (len + inner_fft.get_inplace_scratch_len()),
            fft: inner_fft,
            len,
        }
    }
}

impl<T: DctNum> Dct4<T> for Type4ConvertToFftOdd<T> {
    fn process_dct4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();
        let half_len = len / 2;
        let quarter_len = len / 4;

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        //start by reordering the input into the FFT input
        let mut input_index = half_len;
        let mut fft_index = 0;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[input_index],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        //subtract len to simulate modular arithmetic
        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: -buffer[len - input_index - 1],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: -buffer[input_index],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[len - input_index - 1],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while fft_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[input_index],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        let result_scale = T::SQRT_2() * T::half();
        let second_half_sign = if len % 4 == 1 { T::one() } else { -T::one() };

        //post-process the results 4 at a time
        let mut output_sign = T::one();
        for i in 0..quarter_len {
            let fft_result = fft_buffer[4 * i + 1] * (output_sign * result_scale);
            let next_result = fft_buffer[4 * i + 3] * (output_sign * result_scale);

            buffer[i * 2] = fft_result.re + fft_result.im;
            buffer[i * 2 + 1] = -next_result.re + next_result.im;

            buffer[len - i * 2 - 2] = (next_result.re + next_result.im) * second_half_sign;
            buffer[len - i * 2 - 1] = (fft_result.re - fft_result.im) * second_half_sign;

            output_sign = output_sign.neg();
        }

        //we either have 1 or 3 elements left over that we couldn't get in the above loop, handle them here
        if len % 4 == 1 {
            buffer[half_len] = fft_buffer[0].re * output_sign * result_scale;
        } else {
            let fft_result = fft_buffer[len - 2] * (output_sign * result_scale);

            buffer[half_len - 1] = fft_result.re + fft_result.im;
            buffer[half_len + 1] = -fft_result.re + fft_result.im;
            buffer[half_len] = -fft_buffer[0].re * output_sign * result_scale;
        }
    }
}
impl<T: DctNum> Dst4<T> for Type4ConvertToFftOdd<T> {
    fn process_dst4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let len = self.len();
        let half_len = len / 2;
        let quarter_len = len / 4;

        let complex_scratch = into_complex_mut(scratch);
        let (fft_buffer, fft_scratch) = complex_scratch.split_at_mut(len);

        //start by reordering the input into the FFT input
        let mut input_index = half_len;
        let mut fft_index = 0;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[len - input_index - 1],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        //subtract len to simulate modular arithmetic
        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: -buffer[input_index],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: -buffer[len - input_index - 1],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while input_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[input_index],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        input_index = input_index - len;
        while fft_index < len {
            fft_buffer[fft_index] = Complex {
                re: buffer[len - input_index - 1],
                im: T::zero(),
            };

            input_index += 4;
            fft_index += 1;
        }

        // run the fft
        self.fft.process_with_scratch(fft_buffer, fft_scratch);

        let result_scale = T::SQRT_2() * T::half();
        let second_half_sign = if len % 4 == 1 { T::one() } else { -T::one() };

        //post-process the results 4 at a time
        let mut output_sign = T::one();
        for i in 0..quarter_len {
            let fft_result = fft_buffer[4 * i + 1] * (output_sign * result_scale);
            let next_result = fft_buffer[4 * i + 3] * (output_sign * result_scale);

            buffer[i * 2] = fft_result.re + fft_result.im;
            buffer[i * 2 + 1] = next_result.re - next_result.im;

            buffer[len - i * 2 - 2] = -(next_result.re + next_result.im) * second_half_sign;
            buffer[len - i * 2 - 1] = (fft_result.re - fft_result.im) * second_half_sign;

            output_sign = output_sign.neg();
        }

        //we either have 1 or 3 elements left over that we couldn't get in the above loop, handle them here
        if len % 4 == 1 {
            buffer[half_len] = fft_buffer[0].re * output_sign * result_scale;
        } else {
            let fft_result = fft_buffer[len - 2] * (output_sign * result_scale);

            buffer[half_len - 1] = fft_result.re + fft_result.im;
            buffer[half_len + 1] = -fft_result.re + fft_result.im;
            buffer[half_len] = fft_buffer[0].re * output_sign * result_scale;
        }
    }
}
impl<T: DctNum> RequiredScratch for Type4ConvertToFftOdd<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}
impl<T: DctNum> TransformType4<T> for Type4ConvertToFftOdd<T> {}
impl<T> Length for Type4ConvertToFftOdd<T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::Type4Naive;

    use crate::test_utils::{compare_float_vectors, random_signal};
    use rustfft::FftPlanner;

    /// Verify that our fast implementation of the DCT4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dct4_via_fft_odd() {
        for n in 0..50 {
            let size = 2 * n + 1;
            println!("{}", size);

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            println!("input: {:?}", actual_buffer);

            let naive_dct = Type4Naive::new(size);
            naive_dct.process_dct4(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dct = Type4ConvertToFftOdd::new(fft_planner.plan_fft_forward(size));
            dct.process_dct4(&mut actual_buffer);

            println!("expected: {:?}", expected_buffer);
            println!("actual: {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&actual_buffer, &expected_buffer),
                "len = {}",
                size
            );
        }
    }

    /// Verify that our fast implementation of the DST4 gives the same output as the slow version, for many different inputs
    #[test]
    fn test_dst4_via_fft_odd() {
        for n in 0..50 {
            let size = 2 * n + 1;
            println!("{}", size);

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            println!("input: {:?}", actual_buffer);

            let naive_dst = Type4Naive::new(size);
            naive_dst.process_dst4(&mut expected_buffer);

            let mut fft_planner = FftPlanner::new();
            let dst = Type4ConvertToFftOdd::new(fft_planner.plan_fft_forward(size));
            dst.process_dst4(&mut actual_buffer);

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
