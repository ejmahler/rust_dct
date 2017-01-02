use num::{Zero, One, Signed, FromPrimitive};
use super::dct_type_4::DCT4;

use std::fmt::Debug;

pub struct MDCT<T> {
    dct: super::dct_type_4::DCT4<T>,
    dct_buffer: Vec<T>,
    window: Vec<T>,
}

impl<T> MDCT<T>
    where T: Signed + FromPrimitive + Copy + 'static + Debug
{
    /// Creates a new MDCT context that will process signals of length `len * 2`, resulting in outputs of length `len`
    pub fn new(len: usize) -> Self {
        assert!(len % 2 == 0, "The MDCT `len` parameter must be even");
        MDCT {
            dct: DCT4::new(len),
            dct_buffer: vec![Zero::zero(); len],
            window: vec![One::one(); len*2],
        }
    }

    /// Creates a new MDCT context that will process signals of length `len * 2`, resulting in outputs of length `len`
    pub fn new_windowed<F>(len: usize, window_fn: F) -> Self where F: Fn(usize) -> T {
        assert!(len % 2 == 0, "The MDCT `len` parameter must be even");
        MDCT {
            dct: DCT4::new(len),
            dct_buffer: vec![Zero::zero(); len],
            window: (0..len*2).map(window_fn).collect(),
        }
    }

    /// Runs the MDCT on the input `signal` buffer, and places the output in the
    /// `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

        assert_eq!(signal.len(), self.dct_buffer.len() * 2);
        assert_eq!(spectrum.len(), self.dct_buffer.len());

        let group_size = self.dct_buffer.len() / 2;

        //we're going to scale the input by the window vec, then divide it into four sequential groups: (a,b,c,d)

        //group d is negated and copied to the first half of the dct input
        let window_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale);
        for (element, d) in self.dct_buffer.iter_mut().zip(window_iter.skip(group_size * 3)) {
            *element = -d;
        }

        //group a is copied to the second half of the dct input
        let window_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale);
        for (element, a) in self.dct_buffer.iter_mut().skip(group_size).zip(window_iter) {
            *element = a;
        }

        //groups b and c are reversed then subtracted from the dct input
        let window_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale);
        for (element, bc) in self.dct_buffer.iter_mut().zip(window_iter.rev().skip(group_size)) {
            *element = *element  -bc;
        }

        self.dct.process(&self.dct_buffer, spectrum);
    }

    /// Runs the inverse MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// Note that this *ADDS* the result into the spectrum, rather than directly setting the spectrum
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process_inverse(&mut self, signal: &[T], spectrum: &mut [T]) {

        assert_eq!(signal.len(), self.dct_buffer.len());
        assert_eq!(spectrum.len(), self.dct_buffer.len() * 2);

        self.dct.process(signal, self.dct_buffer.as_mut_slice());

        let group_size = self.dct_buffer.len() / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).zip(self.dct_buffer[group_size..].iter()) {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size).zip(self.dct_buffer[group_size..].iter().rev()) {
            *output = *output - *val * *window_val;;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size*2).zip(self.dct_buffer[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size*3).zip(self.dct_buffer[..group_size].iter()) {
            *output = *output - *val * *window_val;;
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    use ::test_utils::{compare_float_vectors, random_signal};

    /// Verify that our O(N^2) implementation of the MDCT is working as expected
    #[test]
    fn test_slow() {
        let input_list = vec![
            vec![0_f32,0_f32,0_f32,0_f32],
            vec![1_f32,1_f32,-5_f32,5_f32],
            vec![7_f32, 3_f32, 8_f32, 4_f32,-1_f32, 3_f32, 0_f32, 4_f32]
        ];
        let expected_list = vec![
            vec![0_f32, 0_f32],
            vec![0_f32, 0_f32],
            vec![-4.7455063, -2.073643, -2.2964284, 8.479767],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {
            let mut output = vec![0f32; input.len() / 2];
            execute_slow(&input, output.as_mut_slice());

            println!("expected: {:?}, actual: {:?}", expected, output);

            compare_float_vectors(&expected.as_slice(), &output.as_slice());
        }
    }

    /// Verify that our O(N^2) implementation of the IMDCT is working as expected
    /// by running the inverse and then running the forward transform on the result, and verifying that we get what we started with
    #[test]
    fn test_slow_inverse() {
        let input_list = vec![
            vec![0_f32, 0_f32],
            vec![1_f32, 5_f32],
            vec![1_f32, 2_f32, 3_f32, 4_f32]
        ];

        for input in input_list {
            let mut intermediate = vec![0f32; input.len() * 2];
            let mut output = vec![0f32; input.len()];

            execute_slow_inverse(&input, intermediate.as_mut_slice());
            execute_slow(&intermediate, output.as_mut_slice());

            println!("input: {:?}, output: {:?}", input, output);

            compare_float_vectors(&input.as_slice(), &output.as_slice());
        }
    }

    /// Verify that our fast implementation of the MDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for i in 1..11 {
            let size = i * 4;
            let input = random_signal(size);

            let mut slow_output = vec![0f32; size / 2];
            execute_slow(&input, slow_output.as_mut_slice());

            let mut dct = MDCT::new(size / 2);
            let mut fast_output = vec![0f32; size / 2];
            dct.process(&input, fast_output.as_mut_slice());

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }

    /// Verify that our fast implementation of the IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast_inverse() {

        for i in 1..11 {
            let size = i * 2;
            let input = random_signal(size);

            let mut slow_output = vec![0f32; size * 2];
            execute_slow_inverse(&input, slow_output.as_mut_slice());

            let mut dct = MDCT::new(size);
            let mut fast_output = vec![0f32; size * 2];
            dct.process_inverse(&input, fast_output.as_mut_slice());

            //we have to scale the output by 1/size
            for element in fast_output.iter_mut() {
                *element = *element / (size as f32);
            }

            println!("expected: {:?}, actual: {:?}", slow_output, fast_output);

            compare_float_vectors(&slow_output.as_slice(), &fast_output);
        }
    }


/*
    #[test]
    fn test_inverse_overlap() {

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for i in 1..17 {
            let segment_size = i * 4;
            let num_segments = 3;

            let signal = random_signal(segment_size * num_segments);

            //we'll need 2 "garbage" segments at the beginning and end, so num_segments + 2 garbage segments
            let mut signal_buffer = vec![0f32; segment_size * (num_segments + 2)];
            let mut output_buffer = signal_buffer.clone();
            signal_buffer[segment_size..(segment_size*(num_segments + 1))].copy_from_slice(&signal);

            let mut intermediate_buffer = vec![signal_buffer.len() - 1];

            let mut dct = MDCT::new(segment_size);

            //setup is complete, now perform our processing on overlapping segments of the signal buffer
            for (i, spectrum) in intermediate_buffer.chunks_mut().enumerate() {
                let signal = signal_buffer[(segment_size * i)..(segment_size * (i + 2))];
                dct.process(signal, spectrum);
            }

            //perform the inverse, from the 
        }
    }
    */

    fn execute_slow(input: &[f32], output: &mut [f32]) {
        let size_float = output.len() as f32;

        for k in 0..output.len() {
            let mut current_value = 0_f32;

            let k_float = k as f32;

            for n in 0..input.len() {
                let n_float = n as f32;

                current_value +=
                    input[n] * (f32::consts::PI * (n_float + 0.5_f32 + size_float * 0.5) * (k_float + 0.5_f32) / size_float).cos();
            }
            output[k] = current_value;
        }
    }

    fn execute_slow_inverse(input: &[f32], output: &mut [f32]) {
        let size_float = input.len() as f32;

        for n in 0..output.len() {
            let mut current_value = 0_f32;

            let n_float = n as f32;

            for k in 0..input.len() {
                let k_float = k as f32;

                current_value +=
                    input[k] * (f32::consts::PI * (n_float + 0.5_f32 + size_float * 0.5) * (k_float + 0.5_f32) / size_float).cos();
            }
            output[n] += current_value / size_float;
        }
    }
}