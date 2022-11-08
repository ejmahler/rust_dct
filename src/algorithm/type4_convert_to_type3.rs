use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::Length;

use crate::common::dct_error_inplace;
use crate::{twiddles, Dct4, DctNum, Dst4, RequiredScratch, TransformType2And3, TransformType4};

/// DCT4 and DST4 implementation that converts the problem into two DCT3 of half size.
///
/// If the inner DCT3 is O(nlogn), then so is this. This algorithm can only be used if the problem size is even.
///
/// ~~~
/// // Computes a DCT Type 4 of size 1234
/// use std::sync::Arc;
/// use rustdct::Dct4;
/// use rustdct::algorithm::Type4ConvertToType3Even;
/// use rustdct::DctPlanner;
///
/// let len = 1234;
/// let mut planner = DctPlanner::new();
/// let inner_dct3 = planner.plan_dct3(len / 2);
///
/// let dct = Type4ConvertToType3Even::new(inner_dct3);
///
/// let mut buffer = vec![0f32; len];
/// dct.process_dct4(&mut buffer);
/// ~~~
pub struct Type4ConvertToType3Even<T> {
    inner_dct: Arc<dyn TransformType2And3<T>>,
    twiddles: Box<[Complex<T>]>,
    scratch_len: usize,
}

impl<T: DctNum> Type4ConvertToType3Even<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_dct.len() * 2`.
    pub fn new(inner_dct: Arc<dyn TransformType2And3<T>>) -> Self {
        let inner_len = inner_dct.len();
        let len = inner_len * 2;

        let twiddles: Vec<Complex<T>> = (0..inner_len)
            .map(|i| twiddles::single_twiddle(2 * i + 1, len * 8).conj())
            .collect();

        let inner_scratch = inner_dct.get_scratch_len();
        let scratch_len = if inner_scratch <= len {
            len
        } else {
            len + inner_scratch
        };

        Self {
            inner_dct: inner_dct,
            twiddles: twiddles.into_boxed_slice(),
            scratch_len,
        }
    }
}
impl<T: DctNum> Dct4<T> for Type4ConvertToType3Even<T> {
    fn process_dct4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let (self_scratch, extra_scratch) = scratch.split_at_mut(self.len());

        let len = self.len();
        let inner_len = len / 2;

        //pre-process the input by splitting into into two arrays, one for the inner DCT3, and the other for the DST3
        let (mut output_left, mut output_right) = self_scratch.split_at_mut(inner_len);

        output_left[0] = buffer[0] * T::two();
        for k in 1..inner_len {
            output_left[k] = buffer[2 * k - 1] + buffer[2 * k];
            output_right[k - 1] = buffer[2 * k - 1] - buffer[2 * k];
        }
        output_right[inner_len - 1] = buffer[len - 1] * T::two();

        //run the two inner DCTs on our separated arrays
        let inner_scratch = if extra_scratch.len() > 0 {
            extra_scratch
        } else {
            &mut buffer[..]
        };

        self.inner_dct
            .process_dct3_with_scratch(&mut output_left, inner_scratch);
        self.inner_dct
            .process_dst3_with_scratch(&mut output_right, inner_scratch);

        //post-process the data by combining it back into a single array
        for k in 0..inner_len {
            let twiddle = self.twiddles[k];
            let cos_value = output_left[k];
            let sin_value = output_right[k];

            buffer[k] = cos_value * twiddle.re + sin_value * twiddle.im;
            buffer[len - 1 - k] = cos_value * twiddle.im - sin_value * twiddle.re;
        }
    }
}
impl<T: DctNum> Dst4<T> for Type4ConvertToType3Even<T> {
    fn process_dst4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]) {
        let scratch = validate_buffers!(buffer, scratch, self.len(), self.get_scratch_len());

        let (self_scratch, extra_scratch) = scratch.split_at_mut(self.len());

        let len = self.len();
        let inner_len = len / 2;

        //pre-process the input by splitting into into two arrays, one for the inner DCT3, and the other for the DST3
        let (mut output_left, mut output_right) = self_scratch.split_at_mut(inner_len);

        output_right[0] = buffer[0] * T::two();
        for k in 1..inner_len {
            output_left[k - 1] = buffer[2 * k - 1] + buffer[2 * k];
            output_right[k] = buffer[2 * k] - buffer[2 * k - 1];
        }
        output_left[inner_len - 1] = buffer[len - 1] * T::two();

        //run the two inner DCTs on our separated arrays
        let inner_scratch = if extra_scratch.len() > 0 {
            extra_scratch
        } else {
            &mut buffer[..]
        };

        self.inner_dct
            .process_dst3_with_scratch(&mut output_left, inner_scratch);
        self.inner_dct
            .process_dct3_with_scratch(&mut output_right, inner_scratch);

        //post-process the data by combining it back into a single array
        for k in 0..inner_len {
            let twiddle = self.twiddles[k];
            let cos_value = output_left[k];
            let sin_value = output_right[k];

            buffer[k] = cos_value * twiddle.re + sin_value * twiddle.im;
            buffer[len - 1 - k] = sin_value * twiddle.re - cos_value * twiddle.im;
        }
    }
}
impl<T> RequiredScratch for Type4ConvertToType3Even<T> {
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }
}
impl<T: DctNum> TransformType4<T> for Type4ConvertToType3Even<T> {}
impl<T> Length for Type4ConvertToType3Even<T> {
    fn len(&self) -> usize {
        self.twiddles.len() * 2
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algorithm::{Type2And3Naive, Type4Naive};
    use crate::test_utils::{compare_float_vectors, random_signal};

    #[test]
    fn unittest_dct4_via_type3() {
        for inner_size in 1..20 {
            let size = inner_size * 2;

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dct4 = Type4Naive::new(size);
            naive_dct4.process_dct4(&mut expected_buffer);

            let inner_dct3 = Arc::new(Type2And3Naive::new(inner_size));
            let dct = Type4ConvertToType3Even::new(inner_dct3);
            dct.process_dct4(&mut actual_buffer);

            println!("");
            println!("expected: {:?}", expected_buffer);
            println!("actual:   {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&expected_buffer, &actual_buffer),
                "len = {}",
                size
            );
        }
    }

    #[test]
    fn unittest_dst4_via_type3() {
        for inner_size in 1..20 {
            let size = inner_size * 2;

            let mut expected_buffer = random_signal(size);
            let mut actual_buffer = expected_buffer.clone();

            let naive_dst4 = Type4Naive::new(size);
            naive_dst4.process_dst4(&mut expected_buffer);

            let inner_dst3 = Arc::new(Type2And3Naive::new(inner_size));
            let dst = Type4ConvertToType3Even::new(inner_dst3);
            dst.process_dst4(&mut actual_buffer);

            println!("");
            println!("expected: {:?}", expected_buffer);
            println!("actual:   {:?}", actual_buffer);

            assert!(
                compare_float_vectors(&expected_buffer, &actual_buffer),
                "len = {}",
                size
            );
        }
    }
}
