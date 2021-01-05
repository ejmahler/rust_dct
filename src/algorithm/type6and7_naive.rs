use rustfft::Length;

use ::{Dct6, Dct7, Dst6, Dst7, Dct6And7, Dst6And7};
use common;

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
/// let mut dct6_input:  Vec<f32> = vec![0f32; len];
/// let mut dct6_output: Vec<f32> = vec![0f32; len];
/// naive.process_dct6(&mut dct6_input, &mut dct6_output);
/// 
/// let mut dct7_input:  Vec<f32> = vec![0f32; len];
/// let mut dct7_output: Vec<f32> = vec![0f32; len];
/// naive.process_dct7(&mut dct7_input, &mut dct7_output);
/// ~~~
pub struct Dct6And7Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DctNum> Dct6And7Naive<T> {
    /// Creates a new DCT6 and DCT7 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 - 1) as f64;

        let twiddles: Vec<T> = (0..len*4 - 2)
            .map(|i| (constant_factor * (i as f64)).cos())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DctNum> Dct6<T> for Dct6And7Naive<T> {
    fn process_dct6(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        input[input.len() - 1] = input[input.len() - 1] * T::half();
        output[0] = input.iter().fold(T::zero(), |acc, e| acc + *e);

        for k in 1..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2;
            let mut twiddle_index = k;

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
impl<T: common::DctNum> Dct7<T> for Dct6And7Naive<T> {
    fn process_dct7(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        input[0] = input[0] * T::half();

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = input[0];

            let twiddle_stride = k * 2 + 1;
            let mut twiddle_index = twiddle_stride;

            for i in 1..input.len() {
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
impl<T: common::DctNum> Dct6And7<T> for Dct6And7Naive<T>{}
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
/// let mut dst6_input:  Vec<f32> = vec![0f32; len];
/// let mut dst6_output: Vec<f32> = vec![0f32; len];
/// naive.process_dst6(&mut dst6_input, &mut dst6_output);
/// 
/// let mut dst7_input:  Vec<f32> = vec![0f32; len];
/// let mut dst7_output: Vec<f32> = vec![0f32; len];
/// naive.process_dst7(&mut dst7_input, &mut dst7_output);
/// ~~~
pub struct Dst6And7Naive<T> {
    twiddles: Box<[T]>,
}

impl<T: common::DctNum> Dst6And7Naive<T> {
    /// Creates a new DST6 and DST7 context that will process signals of length `len`
    pub fn new(len: usize) -> Self {
        let constant_factor = std::f64::consts::PI / (len * 2 + 1) as f64;

        let twiddles: Vec<T> = (0..len*4 + 2)
            .map(|i| (constant_factor * (i as f64)).sin())
            .map(|c| T::from_f64(c).unwrap())
            .collect();

        Self { twiddles: twiddles.into_boxed_slice() }
    }
}

impl<T: common::DctNum> Dst6<T> for Dst6And7Naive<T> {
    fn process_dst6(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = (k + 1) * 2;
            let mut twiddle_index = k + 1;

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
impl<T: common::DctNum> Dst7<T> for Dst6And7Naive<T> {
    fn process_dst7(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        for k in 0..output.len() {
            let output_cell = output.get_mut(k).unwrap();
            *output_cell = T::zero();

            let twiddle_stride = k * 2 + 1;
            let mut twiddle_index = twiddle_stride;

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
impl<T: common::DctNum> Dst6And7<T> for Dst6And7Naive<T>{}
impl<T> Length for Dst6And7Naive<T> {
    fn len(&self) -> usize {
        (self.twiddles.len() - 2) / 4
    }
}