use rustfft::Length;

use ::{Dct8, Dst8};
use common;

/// Naive O(n^2 ) DCT Type 8 implementation
///
/// ~~~
/// // Computes a naive DCT8 of size 23
/// use rustdct::Dct8;
/// use rustdct::algorithm::Dct8Naive;
///
/// let len = 23;
/// let naive = Dct8Naive::new(len);
/// 
/// let mut dct8_input:  Vec<f32> = vec![0f32; len];
/// let mut dct8_output: Vec<f32> = vec![0f32; len];
/// naive.process_dct8(&mut dct8_input, &mut dct8_output);
/// ~~~
pub struct Dct8Naive<T> {
    twiddles: Box<[T]>,
}
impl<T: common::DctNum> Dct8Naive<T> {
    /// Creates a new DCT8 and DST8 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 + 1) as f64;

        let twiddles: Vec<T> = (0..len*4 + 2)
            .map(|i| (constant_factor * (i as f64 + 0.5)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}
impl<T: common::DctNum> Dct8<T> for Dct8Naive<T> {
    fn process_dct8(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

            for i in 0..input.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + input[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> Length for Dct8Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() - 2) / 4
    }
}

/// Naive O(n^2 ) DST Type 8 implementation
///
/// ~~~
/// // Computes a naive DST8 of size 23
/// use rustdct::Dst8;
/// use rustdct::algorithm::Dst8Naive;
///
/// let len = 23;
/// let naive = Dst8Naive::new(len);
/// 
/// let mut dst8_input:  Vec<f32> = vec![0f32; len];
/// let mut dst8_output: Vec<f32> = vec![0f32; len];
/// naive.process_dst8(&mut dst8_input, &mut dst8_output);
/// ~~~
pub struct Dst8Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DctNum> Dst8Naive<T> {
    /// Creates a new DCT8 and DST8 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 - 1) as f64;

        let twiddles: Vec<T> = (0..len*4 - 2)
            .map(|i| (constant_factor * (i as f64 + 0.5)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DctNum> Dst8<T> for Dst8Naive<T> {
    fn process_dst8(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        input[input.len() - 1] = input[input.len() - 1] * T::half();

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let mut twiddle_index = k;
            let twiddle_stride = k * 2 + 1;

            for i in 0..input.len() {
                let twiddle = self.twiddles[twiddle_index];

                *output_cell = *output_cell + input[i] * twiddle;

                twiddle_index += twiddle_stride;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }
}
impl<T> Length for Dst8Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() + 2) / 4
    }
}
