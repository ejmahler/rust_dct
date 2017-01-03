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
    pub fn new_windowed<F>(len: usize, window_fn: F) -> Self where F: Fn(usize) -> Vec<T> {
        assert!(len % 2 == 0, "The MDCT `len` parameter must be even");

        MDCT {
            dct: DCT4::new(len),
            dct_buffer: vec![Zero::zero(); len],
            window: window_fn(len * 2),
        }
    }

    /// Runs the MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process(&mut self, signal: &[T], spectrum: &mut [T]) {

        assert_eq!(signal.len(), self.dct_buffer.len() * 2);
        assert_eq!(spectrum.len(), self.dct_buffer.len());

        let group_size = self.dct_buffer.len() / 2;

        //we're going to scale the input by the window vec, then divide it into four sequential groups: (a,b,c,d)
        //then rewrite them as TWO segments: (-D-Cr, A-Br) where R means reversed

        //group d is negated and copied to the first half of the dct input

        let group_d_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale).skip(group_size * 3);
        for (element, d) in self.dct_buffer.iter_mut().zip(group_d_iter) {
            *element = -d;
        }

        //group a is copied to the second half of the dct input
        let group_a_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale);
        for (element, a) in self.dct_buffer.iter_mut().skip(group_size).zip(group_a_iter) {
            *element = a;
        }

        //groups b and c are reversed then subtracted from the dct input
        let group_bc_iter = signal.iter().zip(&self.window).map(|(val, scale)| *val * *scale).rev().skip(group_size);
        for (element, bc) in self.dct_buffer.iter_mut().zip(group_bc_iter) {
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
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size*2).zip(self.dct_buffer[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size*3).zip(self.dct_buffer[..group_size].iter()) {
            *output = *output - *val * *window_val;
        }
    }






    /// Runs the MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// This processes several segments of size 'len' as a stream, rather than just a single segment, and unlike 
    /// the single segment version, the output of this function is independently invertible
    ///
    /// # Panics
    /// This method will panic if `signal` or `spectrum` is not a multiple of the specified `segment_len`, or if signal.len() + segment_len != spectrum.len()
    pub fn process_overlapped(&mut self, signal: &[T], spectrum: &mut [T]) {
        let segment_len = self.dct_buffer.len();
        let signal_len = signal.len();
        let spectrum_len = spectrum.len();

        let num_segments = spectrum_len / segment_len;

        assert!(signal_len % segment_len == 0, "The signal size and spectrum size must be a multiple of segment_len");
        assert!(signal_len + segment_len == spectrum_len);

        //first, we need to treat the beginning of the spectrum separately, since there is nothing before it to overlap it with
        self.process_overlap_beginning(&signal[..segment_len], &mut spectrum[..segment_len]);

        //now we can do the middle segments directly from the signal into the spectrum
        for (index, output) in spectrum[segment_len..].chunks_mut(segment_len).take(num_segments - 2).enumerate() {
            let input = &signal[segment_len*index..segment_len*(index+2)];

            self.process(&input, output);
        }

        //finally, we need to treat the end of the spectrum separately, since there is nothing after it to overlap it with
        self.process_overlap_end(&signal[signal_len-segment_len..], &mut spectrum[spectrum_len-segment_len..]);
    }


    /// Runs the inverse overlapped MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// This processes several segments of size 'len' as a stream, rather than just a single segment
    ///
    /// # Panics
    /// This method will panic if `signal` or `spectrum` is not a multiple of the specified `segment_len`, or if signal.len() - segment_len != spectrum.len()
    pub fn process_inverse_overlapped(&mut self, signal: &[T], spectrum: &mut [T]) {
        let segment_len = self.dct_buffer.len();
        let signal_len = signal.len();
        let spectrum_len = spectrum.len();

        let num_segments = signal_len / segment_len;

        assert!(signal_len % segment_len == 0, "The signal size and spectrum size must be a multiple of segment_len");
        assert!(signal_len - segment_len == spectrum_len);

        //first, we need to treat the beginning of the spectrum separately, since there is nothing before it to overlap it with
        self.process_inverse_overlap_beginning(&signal[..segment_len], &mut spectrum[..segment_len]);

        //now we can do the middle segments directly from the signal into the spectrum
        for (index, input) in signal[segment_len..].chunks(segment_len).take(num_segments - 2).enumerate(){
            let output = &mut spectrum[segment_len*index..segment_len*(index+2)];

            self.process_inverse(&input, output);
        }


        //finally, we need to treat the end of the spectrum separately, since there is nothing after it to overlap it with
        self.process_inverse_overlap_end(&signal[signal_len-segment_len..], &mut spectrum[spectrum_len-segment_len..]);
    }





    /// Runs the MDCT on the signal and puts int in the spectrum/. this is specifically intended for the first segment of an overlapped signal
    fn process_overlap_beginning(&mut self, signal: &[T], spectrum: &mut [T]) {
        let group_size = self.dct_buffer.len() / 2;

        //we're going to scale the input by the window vec, then divide it into four sequential groups: (a,b,c,d)
        //then rewrite them as TWO segments: (-D-Cr, A-Br) where R means reversed
        //EXCEPT this is at the beginning of an overlapped segment, which means that A and B don't exist
        //so we're going to treat B as a negative, reversed version of C, and A as a negative, reversed version of D

        //scale the signal by our window function and temporarily put the result in the spectrum
        for (spectrum_val, (signal_val, window_val)) in spectrum.iter_mut().zip(signal.iter().zip(&self.window[self.dct_buffer.len()..])) {
            *spectrum_val = *signal_val * *window_val;
        }

        //we need a scope here in order to contain the spectrum borrow, because we have to borrow it again at the end of the function
        {
            //the spectrum now contains the window-scaled input, we'll now copy that to various places
            let group_c_iter = spectrum[..group_size].iter();
            let group_c_rev_iter = spectrum[..group_size].iter().rev();
            let group_d_iter = spectrum[group_size..].iter();
            let group_d_rev_iter = spectrum[group_size..].iter().rev();

            //the first half of the dct input will be -Cr - D
            for (element, (d_val, cr_val)) in self.dct_buffer.iter_mut().zip(group_d_iter.zip(group_c_rev_iter)) {
                *element = -*cr_val - *d_val;
            }

            //the second half of the dct input should be A - Br, but A and br don't exist. so instead we'll use C - Dr
            for (element, (c_val, dr_val)) in self.dct_buffer[group_size..].iter_mut().zip(group_c_iter.zip(group_d_rev_iter)) {
                *element = *c_val - *dr_val;
            }
        }

        self.dct.process(&self.dct_buffer, spectrum);
    }

    /// Runs the MDCT on the signal and puts int in the spectrum/. this is specifically intended for the final segment of an overlapped signal
    fn process_overlap_end(&mut self, signal: &[T], spectrum: &mut [T]) {
        let group_size = self.dct_buffer.len() / 2;

        //we're going to scale the input by the window vec, then divide it into four sequential groups: (a,b,c,d)
        //then rewrite them as TWO segments: (-D-Cr, A-Br) where R means reversed
        //EXCEPT this is at the end  of an overlapped segment, which means that C and D don't exist
        //so we're going to treat B as a negative, reversed version of C, and A as a negative, reversed version of D

        //scale the signal by our window function and temporarily put the result in the spectrum
        for (spectrum_val, (signal_val, window_val)) in spectrum.iter_mut().zip(signal.iter().zip(&self.window)) {
            *spectrum_val = *signal_val * *window_val;
        }

        //we need a scope here in order to contain the spectrum borrow, because we have to borrow it again at the end of the function
        {
            //the spectrum now contains the window-scaled input, we'll now copy that to various places
            let group_a_iter = spectrum[..group_size].iter();
            let group_a_rev_iter = spectrum[..group_size].iter().rev();
            let group_b_iter = spectrum[group_size..].iter();
            let group_b_rev_iter = spectrum[group_size..].iter().rev();

            //the first half of the dct input should be -Cr - D, but c and D don't exist, so we'll use B+ar instead
            for (element, (b_val, ar_val)) in self.dct_buffer.iter_mut().zip(group_b_iter.zip(group_a_rev_iter)) {
                *element = *b_val + *ar_val;
            }

            //the second half of the dct input should be is A - Br
            for (element, (a_val, br_val)) in self.dct_buffer[group_size..].iter_mut().zip(group_a_iter.zip(group_b_rev_iter)) {
                *element = *a_val - *br_val;
            }
        }

        self.dct.process(&self.dct_buffer, spectrum);
    }





    /// Runs the MDCT on the signal and puts int in the spectrum/. this is specifically intended for the first segment of an overlapped signal
    fn process_inverse_overlap_beginning(&mut self, signal: &[T], spectrum: &mut [T]) {
        self.dct.process(signal, self.dct_buffer.as_mut_slice());

        let group_size = self.dct_buffer.len() / 2;

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter().skip(group_size*2)).zip(self.dct_buffer[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in spectrum.iter_mut().skip(group_size).zip(self.window.iter().skip(group_size*3)).zip(self.dct_buffer[..group_size].iter()) {
            *output = *output - *val * *window_val;
        }
    }

    /// Runs the MDCT on the signal and puts int in the spectrum/. this is specifically intended for the final segment of an overlapped signal
    fn process_inverse_overlap_end(&mut self, signal: &[T], spectrum: &mut [T]) {
        self.dct.process(signal, self.dct_buffer.as_mut_slice());

        let group_size = self.dct_buffer.len() / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).zip(self.dct_buffer[group_size..].iter()) {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(self.window.iter()).skip(group_size).zip(self.dct_buffer[group_size..].iter().rev()) {
            *output = *output - *val * *window_val;
        }
    }
}

pub mod window_fn {
    use num::{Float, FromPrimitive};
    use num::traits::FloatConst;

    pub fn mp3<T>(len: usize) -> Vec<T>
        where T: Float + FloatConst + FromPrimitive
    {
        let constant_term: T = T::PI() / FromPrimitive::from_usize(len).unwrap();
        let half: T = FromPrimitive::from_f32(0.5f32).unwrap();

        (0..len).map(|n| {
            let n_float: T = FromPrimitive::from_usize(n).unwrap();
            (constant_term * (n_float + half)).sin()
        }).collect()
    }

    pub fn vorbis<T>(len: usize) -> Vec<T>
        where T: Float + FloatConst + FromPrimitive
    {
        let constant_term: T = T::PI() / FromPrimitive::from_usize(len).unwrap();
        let half: T = FromPrimitive::from_f32(0.5f32).unwrap();

        (0..len).map(|n| {
            let n_float: T = FromPrimitive::from_usize(n).unwrap();
            let inner_sin = (constant_term * (n_float + half)).sin();

            (T::FRAC_PI_2() * inner_sin * inner_sin).sin()
        }).collect()
    }
}





#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    use ::test_utils::{compare_float_vectors, random_signal, fuzzy_cmp};

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

            for element in output.iter_mut() {
                *element = *element / (input.len() as f32);
            }

            println!("input: {:?}, output: {:?}", input, output);

            compare_float_vectors(&input.as_slice(), &output.as_slice());
        }
    }

    /// Verify that our fast implementation of the MDCT and IMDCT gives the same output as the slow version, for many different inputs
    #[test]
    fn test_fast() {
        for i in 1..11 {
            let size = i * 4;
            let input = random_signal(size);


            //compute the forward direction
            let mut slow_output = vec![0f32; size / 2];
            execute_slow(&input, slow_output.as_mut_slice());

            let mut dct = MDCT::new(size / 2);
            let mut fast_output = vec![0f32; size / 2];
            dct.process(&input, fast_output.as_mut_slice());

            println!("forward expected: {:?}, actual: {:?}", slow_output, fast_output);
            compare_float_vectors(&slow_output, &fast_output);

            //now compute the inverse
            let mut slow_inverse = vec![0f32; size];
            execute_slow_inverse(&slow_output, slow_inverse.as_mut_slice());

            let mut fast_inverse = vec![0f32; size];
            dct.process_inverse(&slow_output, fast_inverse.as_mut_slice());

            println!("inverse expected: {:?}, actual: {:?}", slow_inverse, fast_inverse);
            compare_float_vectors(&slow_inverse, &fast_inverse);
        }
    }

    /// Verify that our fast implementation of the MDCT and IMDCT works when a window function is used
    #[test]
    fn test_fast_windowed() {
        for i in 1..11 {
            let size = i * 4;
            let input = random_signal(size);

            let evaluated_window: Vec<f32> = window_fn::mp3(size);
            
            //first do the forward direction
            let mut dct = MDCT::new_windowed(size / 2, window_fn::mp3);
            let mut fast_output = vec![0f32; size / 2];
            dct.process(&input, fast_output.as_mut_slice());

            //to simulate the window function for the slow algorithm, we'll pre-multiply the input vector with the window function
            let slow_input: Vec<f32> = input.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val).collect();
            let mut slow_output = vec![0f32; size / 2];
            execute_slow(&slow_input, slow_output.as_mut_slice());

            println!("expected forward: {:?}, actual: {:?}", slow_output, fast_output);
            compare_float_vectors(&slow_output, &fast_output);




            //now compute the inverse
            let mut slow_inverse = vec![0f32; size];
            execute_slow_inverse(&slow_output, slow_inverse.as_mut_slice());

            //to simulate the window function for the slow algorithm for the inverse, we'll post-multiply the slow output vector with the window function
            slow_inverse = slow_inverse.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val).collect();

            let mut fast_inverse = vec![0f32; size];
            dct.process_inverse(&slow_output, fast_inverse.as_mut_slice());

            println!("expected inverse: {:?}, actual: {:?}", slow_inverse, fast_inverse);
            compare_float_vectors(&slow_inverse, &fast_inverse);
        }
    }


    /// Verify the TDAC property of the process_overalpped and inverse functions
    #[test]
    fn test_tdac() {

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for num_segments in 1..5 {
            for i in 1..9 {
                let segment_size = i * 2;

                let signal = random_signal(segment_size * num_segments);

                let mut output_buffer = vec![0f32; signal.len()];
                let mut intermediate_buffer = vec![0f32; signal.len() + segment_size];
                let mut dct = MDCT::new(segment_size);

                dct.process_overlapped(&signal, intermediate_buffer.as_mut_slice());
                dct.process_inverse_overlapped(&intermediate_buffer, output_buffer.as_mut_slice());

                //the result isn't normalized, so if we want to compare it to the input signal, we have to normalize it
                for element in output_buffer.iter_mut() {
                    *element = *element / (segment_size as f32);
                }

                println!("num_segments: {}, segment_size: {}, expected: {:?}, actual: {:?}", num_segments, segment_size, signal, output_buffer);

                compare_float_vectors(&signal, &output_buffer);
            }
        }
    }

    /// Verify the TDAC property of the process_overalpped and inverse functions, when a window function is used
    #[test]
    fn test_tdac_windowed() {

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for num_segments in 1..5 {
            for i in 1..9 {
                let segment_size = i * 2;
                let signal = random_signal(segment_size * num_segments);

                let mut output_buffer = vec![0f32; signal.len()];
                let mut intermediate_buffer = vec![0f32; signal.len() + segment_size];

                let mut dct = MDCT::new_windowed(segment_size, window_fn::mp3);

                dct.process_overlapped(&signal, intermediate_buffer.as_mut_slice());
                dct.process_inverse_overlapped(&intermediate_buffer, output_buffer.as_mut_slice());

                for element in output_buffer.iter_mut() {
                    *element = *element * 2f32 / (segment_size as f32);
                }

                println!("num_segments: {}, segment_size: {}, expected: {:?}, actual: {:?}", num_segments, segment_size, signal, output_buffer);

                compare_float_vectors(&signal, &output_buffer);
            }
        }
    }

    /// The beginning and end of the overlapping "stream" are treated with special cases, verify them here
    #[test]
    fn test_overlap_ends() {

        let input_list = vec![
            vec![0_f32, 0_f32],
            vec![2_f32, 10_f32],
            vec![1_f32, 5_f32, 10_f32, 50_f32]
        ];

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for signal in input_list {

            let segment_size = signal.len();

            //compute the fast algorithm into the fast_output
            let mut dct = MDCT::new(segment_size);
            let mut fast_output = vec![0f32; segment_size * 2];
            dct.process_overlapped(&signal, fast_output.as_mut_slice());

            //for the slow algorithm, we have to cmput ethe two slow segments separately
            let mut first_slow_input = vec![0f32; segment_size * 2];
            let mut first_slow_output = vec![0f32; segment_size];
            for (input, signal) in first_slow_input.iter_mut().zip(signal.iter().rev()) {
                *input = -*signal;
            }
            for (input, signal) in first_slow_input.iter_mut().skip(segment_size).zip(signal.iter()) {
                *input = *signal;
            }
            execute_slow(&first_slow_input, first_slow_output.as_mut_slice());

            //second half of slow algorithm
            let mut second_slow_input = vec![0f32; segment_size * 2];
            let mut second_slow_output = vec![0f32; segment_size];
            for (input, signal) in second_slow_input.iter_mut().zip(signal.iter()) {
                *input = *signal;
            }
            for (input, signal) in second_slow_input.iter_mut().skip(segment_size).zip(signal.iter().rev()) {
                *input = -*signal;
            }
            execute_slow(&second_slow_input, second_slow_output.as_mut_slice());

            compare_float_vectors(&first_slow_output, &fast_output[..segment_size]);
            compare_float_vectors(&second_slow_output, &fast_output[segment_size..]);


            //now compute the inverse
            let mut fast_inverse = vec![0f32; segment_size];
            dct.process_inverse_overlapped(&fast_output, fast_inverse.as_mut_slice());

            //for the slow algorithm, we have to cmput ethe two slow segments separately
            let mut slow_inverse = vec![0f32; segment_size * 3];

            execute_slow_inverse(&first_slow_output, &mut slow_inverse[..segment_size*2]);
            execute_slow_inverse(&second_slow_output, &mut slow_inverse[segment_size..]);

            println!("expected: {:?}, actual: {:?}", &slow_inverse[segment_size..segment_size*2], fast_inverse);

            compare_float_vectors(&slow_inverse[segment_size..segment_size*2], &fast_inverse);
        }
    }





    /// The beginning and end of the overlapping "stream" are treated with special cases, verify that they work when a window function is used
    #[test]
    fn test_overlap_ends_windowed() {

        let input_list = vec![
            vec![0_f32, 0_f32],
            vec![1_f32, 100_f32],
            vec![1_f32, 5_f32, 10_f32, 50_f32]
        ];

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for signal in input_list {

            let segment_size = signal.len();
            let evaluated_window: Vec<f32> = window_fn::mp3(segment_size * 2);

            //compute the fast algorithm into the fast_output
            let mut dct = MDCT::new_windowed(segment_size, window_fn::mp3);
            let mut fast_output = vec![0f32; segment_size * 2];
            dct.process_overlapped(&signal, fast_output.as_mut_slice());


            //for the slow algorithm, we have to cmput ethe two slow segments separately
            let mut first_slow_input = vec![0f32; segment_size * 2];
            let mut first_slow_output = vec![0f32; segment_size];
            for (input, signal) in first_slow_input.iter_mut().zip(signal.iter().rev()) {
                *input = -*signal;
            }
            for (input, signal) in first_slow_input.iter_mut().skip(segment_size).zip(signal.iter()) {
                *input = *signal;
            }
            first_slow_input = first_slow_input.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val).collect();
            execute_slow(&first_slow_input, first_slow_output.as_mut_slice());


            //second half of slow algorithm
            let mut second_slow_input = vec![0f32; segment_size * 2];
            let mut second_slow_output = vec![0f32; segment_size];
            for (input, signal) in second_slow_input.iter_mut().zip(signal.iter()) {
                *input = *signal;
            }
            for (input, signal) in second_slow_input.iter_mut().skip(segment_size).zip(signal.iter().rev()) {
                *input = -*signal;
            }
            second_slow_input = second_slow_input.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val).collect();
            execute_slow(&second_slow_input, second_slow_output.as_mut_slice());

            compare_float_vectors(&first_slow_output, &fast_output[..segment_size]);
            compare_float_vectors(&second_slow_output, &fast_output[segment_size..]);



            //now compute the inverse via bot the fast and slow methods
            let mut fast_inverse = vec![0f32; segment_size];
            dct.process_inverse_overlapped(&fast_output, fast_inverse.as_mut_slice());

            //for the slow algorithm, we have to cmput ethe two slow segments separately
            let mut first_slow_inverse = vec![0f32; segment_size * 2];
            let mut second_slow_inverse = vec![0f32; segment_size * 2];

            execute_slow_inverse(&first_slow_output, &mut first_slow_inverse);
            execute_slow_inverse(&second_slow_output, &mut second_slow_inverse);

            //apply the window function to both the slow outputs
            let first_slow_inverse_mapped = first_slow_inverse.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val);
            let second_slow_inverse_mapped = second_slow_inverse.iter().zip(&evaluated_window).map(|(element, window_val)| *element * *window_val);
            let slow_inverse: Vec<f32> = first_slow_inverse_mapped.skip(segment_size).zip(second_slow_inverse_mapped).map(|(first, second)| first + second).collect();

            compare_float_vectors(&slow_inverse, &fast_inverse);
        }
    }

    /// Verify that each of the built-in window functions does what we expect
    #[test]
    fn test_window_fns() {
        for test_fn in &[window_fn::mp3, window_fn::vorbis] {
            for half_size in 1..20 {
                let evaluated_window: Vec<f32> = test_fn(half_size * 2);

                //verify that for all i from 0 to half_size, window[i]^2 + window[i+half_size]^2 == 1
                //also known as the "Princen-Bradley condition"
                for i in 0..half_size {
                    let first = evaluated_window[i];
                    let second = evaluated_window[i + half_size];
                    assert!(fuzzy_cmp(first*first + second*second, 1f32, 0.001f32));
                }
            }
        }
    }






    

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
            output[n] += current_value;
        }
    }
}