use std::sync::Arc;

use rustfft::num_traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{FFT, Length};

use common;
use ::{DST6, DST7, DST6And7};

/// DST6 and DST7 implementation that converts the problem into a FFT of the same size
///
/// ~~~
/// // Computes a O(NlogN) DST6 and DST7 of size 1234 by converting them to FFTs
/// use rustdct::{DST6, DST7};
/// use rustdct::algorithm::DST6And7ConvertToFFT;
/// use rustdct::rustfft::FFTplanner;
///
/// let len = 1234;
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(len * 2 + 1);
///
/// let dct = DST6And7ConvertToFFT::new(fft);
/// 
/// let mut dst6_input:  Vec<f32> = vec![0f32; len];
/// let mut dst6_output: Vec<f32> = vec![0f32; len];
/// dct.process_dst6(&mut dst6_input, &mut dst6_output);
/// 
/// let mut dst7_input:  Vec<f32> = vec![0f32; len];
/// let mut dst7_output: Vec<f32> = vec![0f32; len];
/// dct.process_dst7(&mut dst7_input, &mut dst7_output);
/// ~~~
pub struct DST6And7ConvertToFFT<T> {
    inner_fft: Arc<dyn FFT<T>>,
}

impl<T: common::DCTnum> DST6And7ConvertToFFT<T> {
    /// Creates a new DST6 and DST7 context that will process signals of length `(inner_fft.len() - 1) / 2`.
    pub fn new(inner_fft: Arc<dyn FFT<T>>) -> Self {
        assert!(inner_fft.len() % 2 == 1, "The 'DST6And7ConvertToFFT' algorithm requires an odd-len FFT. Provided len={}", inner_fft.len());
        assert!(!inner_fft.is_inverse(), "The 'DST6And7ConvertToFFT' algorithm requires a forward FFT, but an inverse FFT was provided");

        Self { inner_fft }
    }
}
impl<T: common::DCTnum> DST6<T> for DST6And7ConvertToFFT<T> {
    fn process_dst6(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let mut fft_buffer = vec![Complex::zero(); self.inner_fft.len() * 2];
        let (fft_input, fft_output) = fft_buffer.split_at_mut(self.inner_fft.len());

        // Copy the input to the odd imaginary components of the FFT inputs
        for i in 0..input.len() {
            fft_input[i * 2 + 1].im = input[i];
        }

        // inner fft
        self.inner_fft.process(fft_input, fft_output);

        // Copy the first half of the array to the odd-indexd elements
        let even_count = (input.len() + 1) / 2;
        let odd_count = input.len() - even_count;
        for i in 0..odd_count {
            let output_index = i * 2 + 1;
            output[output_index] = fft_output[i + 1].re;
        }

        // Copy the second half of the array to the reversed even-indexed elements
        for i in 0..even_count {
            let output_index = 2 * (even_count - i - 1);
            output[output_index] = fft_output[i + odd_count + 1].re;
        }
    }
}
impl<T: common::DCTnum> DST7<T> for DST6And7ConvertToFFT<T> {
    fn process_dst7(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let mut fft_buffer = vec![Complex::zero(); self.inner_fft.len() * 2];
        let (fft_input, fft_output) = fft_buffer.split_at_mut(self.inner_fft.len());

        // Copy all the even-indexed elements to the back of the FFT input array
        let even_count = (input.len() + 1) / 2;
        for i in 0..even_count {
            let input_index = i * 2;
            let inner_index = input.len() + 1 + i;
            fft_input[inner_index] = Complex{ re: input[input_index], im: T::zero() };
        }
        // Copy all the odd-indexed elements in reverse order 
        let odd_count = input.len() - even_count;
        for i in 0..odd_count {
            let input_index = 2 * (odd_count - i) - 1;
            let inner_index = input.len() + even_count + 1 + i;
            fft_input[inner_index] = Complex{ re: input[input_index], im: T::zero() };
        }
        // Copy the back of the array to the front, negated and reversed
        for i in 0..input.len() {
            fft_input[i + 1] = -fft_input[fft_input.len() - 1 - i];
        }

        // inner fft
        self.inner_fft.process(fft_input, fft_output);

        // copy output back
        for i in 0..output.len() {
            output[i] = fft_output[i * 2 + 1].im * T::half();
        }
    }
}
impl<T: common::DCTnum> DST6And7<T> for DST6And7ConvertToFFT<T>{}
impl<T> Length for DST6And7ConvertToFFT<T> {
    fn len(&self) -> usize {
        (self.inner_fft.len() - 1) / 2
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use algorithm::DST6And7Naive;

    use test_utils::{compare_float_vectors, random_signal};
    use rustfft::FFTplanner;

    /// Verify that our fast implementation of the DCT6 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dst6_via_fft() {
        for size in 2..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let naive_dst = DST6And7Naive::new(size);
            naive_dst.process_dst6(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let dst = DST6And7ConvertToFFT::new(fft_planner.plan_fft(size * 2 + 1));
            dst.process_dst6(&mut actual_input, &mut actual_output);

            println!("{}", size);
            println!("expected: {:?}", expected_output);
            println!("actual: {:?}", actual_output);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }

    /// Verify that our fast implementation of the DST7 gives the same output as the naive version, for many different inputs
    #[test]
    fn test_dst7_via_fft() {
        for size in 2..20 {
            let mut expected_input = random_signal(size);
            let mut actual_input = random_signal(size);

            let mut expected_output = vec![0f32; size];
            let mut actual_output = vec![0f32; size];

            let naive_dst = DST6And7Naive::new(size);
            naive_dst.process_dst7(&mut expected_input, &mut expected_output);

            let mut fft_planner = FFTplanner::new(false);
            let dst = DST6And7ConvertToFFT::new(fft_planner.plan_fft(size * 2 + 1));
            dst.process_dst7(&mut actual_input, &mut actual_output);

            println!("{}", size);
            println!("expected: {:?}", expected_output);
            println!("actual:   {:?}", actual_output);

            assert!(
                compare_float_vectors(&actual_output, &expected_output),
                "len = {}",
                size
            );
        }
    }
}
