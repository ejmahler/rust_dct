use num::{Zero, One};

use dct4::DCT4;
use plan::DCTPlanner;
use DCTnum;

pub struct MDCT<T> {
    dct: Box<DCT4<T>>,
    dct_buffer: Vec<T>,
    window: Vec<T>,
}

impl<T: DCTnum> MDCT<T> {
    /// Creates a new MDCT context that will process signals of length `len * 2`, resulting in outputs of length `len`
    pub fn new(len: usize) -> Self {
        assert!(len % 2 == 0, "The MDCT `len` parameter must be even");

        let mut planner = DCTPlanner::new();
        MDCT {
            dct: planner.plan_dct4(len),
            dct_buffer: vec![Zero::zero(); len],
            window: vec![One::one(); len*2],
        }
    }

    /// Creates a new MDCT context that will process signals of length `len * 2`, resulting in outputs of length `len`
    pub fn new_windowed<F>(len: usize, window_fn: F) -> Self where F: Fn(usize) -> Vec<T> {
        assert!(len % 2 == 0, "The MDCT `len` parameter must be even");

        let mut planner = DCTPlanner::new();
        MDCT {
            dct: planner.plan_dct4(len),
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

        let (left, right) = signal.split_at(signal.len() / 2);
        self.process_internal(left, right, spectrum);
    }

    /// Runs the inverse MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// Note that this *ADDS* the result into the spectrum, rather than directly setting the spectrum
    ///
    /// # Panics
    /// This method will panic if `signal` and `spectrum` are not the length
    /// specified in the struct's constructor.
    pub fn process_inverse(&mut self, signal: &[T], spectrum: &mut [T]) {
        let spectrum_len = spectrum.len();

        assert_eq!(signal.len(), self.dct_buffer.len());
        assert_eq!(spectrum_len, self.dct_buffer.len() * 2);

        let (left, right) = spectrum.split_at_mut(spectrum_len / 2);
        self.process_inverse_internal(signal, left, right);
    }






    /// Runs the MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// This processes several segments of size 'len' as a stream, rather than just a single segment, and unlike 
    /// the single segment version, the output of this function is independently invertible
    ///
    /// # Panics
    /// This method will panic if `signal` or `spectrum` is not a multiple of the specified `segment_len`, or if signal.len() != spectrum.len()
    pub fn process_overlapped(&mut self, signal: &[T], spectrum: &mut [T]) {
        let segment_len = self.dct_buffer.len();
        let signal_len = signal.len();

        assert!(signal_len % segment_len == 0, "The signal size and spectrum size must be a multiple of the `len` provided in the constructor");
        assert!(signal_len == spectrum.len());

        let num_segments = signal_len / segment_len;

        //do all of the "normal" segments
        for (index, output) in spectrum.chunks_mut(segment_len).take(num_segments - 1).enumerate() {
            let input_slice = &signal[index*segment_len..(index+2)*segment_len];
            let (input_left, input_right) = input_slice.split_at(segment_len);

            self.process_internal(input_left, input_right, output);
        }

        //finally, we need to treat the end of the signal separately
        //the beginning has nothing to overlap before it, and the end has nothing to overlap after it
        //so our final result, to handle both the beginning and end, will be to "wrap" the end around to the beginning
        self.process_internal(&signal[signal_len-segment_len..], &signal[..segment_len], &mut spectrum[signal_len-segment_len..]);
    }


    /// Runs the inverse overlapped MDCT on the input `signal` buffer, and places the output in the `spectrum` buffer.
    /// This processes several segments of size 'len' as a stream, rather than just a single segment
    ///
    /// # Panics
    /// This method will panic if `signal` or `spectrum` is not a multiple of the specified `segment_len`, or if signal.len() != spectrum.len()
    pub fn process_inverse_overlapped(&mut self, signal: &[T], spectrum: &mut [T]) {
        let segment_len = self.dct_buffer.len();
        let signal_len = signal.len();

        let num_segments = signal_len / segment_len;

        assert!(signal_len % segment_len == 0, "The signal size and spectrum size must be a multiple of the `len` provided in the constructor");
        assert!(signal_len == spectrum.len());

        if num_segments == 1 {
            //when there's only a single segment, we can't get different slices of the beginning and end because they're the same thing
            //so we have to call a special method
            self.process_inverse_internal_single(signal, spectrum);
        } else {

            //do each of the "normal_segments"
            for (index, input) in signal.chunks(segment_len).take(num_segments - 1).enumerate() {
                let output_slice = &mut spectrum[index*segment_len..(index+2)*segment_len];
                let (output_left, output_right) = output_slice.split_at_mut(segment_len);

                self.process_inverse_internal(input, output_left, output_right);
            }

            //finally, treat the final segment differently. it contains data for the final output spegment, followed by the first output segment
            let (spectrum_main, spectrum_end) = spectrum.split_at_mut(signal_len - segment_len);
            self.process_inverse_internal(&signal[signal_len-segment_len..], spectrum_end, &mut spectrum_main[..segment_len]);
        }
    }





    /// Runs the MDCT on the signal and puts int in the spectrum. we split the signal into two halves to make code elsewhere in this impl simpler
    /// The two halves are usually contiguous, but sometimes not
    fn process_internal(&mut self, signal_left: &[T], signal_right: &[T], spectrum: &mut [T]) {
        let segment_size = self.dct_buffer.len();
        let group_size = segment_size / 2;


        //we're going to divide signal_left into two subgroups, (a,b), and signal_right into two subgroups: (c,d)
        //then scale them by the window function, then combine them into two subgroups: (-D-Cr, A-Br) where R means reversed
        let group_a_iter =     signal_left.iter().zip(&self.window).map(|(a, window_val)| *a * *window_val).take(group_size);
        let group_b_rev_iter = signal_left.iter().zip(&self.window).map(|(b, window_val)| *b * *window_val).rev().take(group_size);
        let group_c_rev_iter = signal_right.iter().zip(&self.window[segment_size..]).map(|(c, window_val)| *c * *window_val).rev().skip(group_size);
        let group_d_iter =     signal_right.iter().zip(&self.window[segment_size..]).map(|(d, window_val)| *d * *window_val).skip(group_size);

        //the first half of the dct input is -Cr - D
        for (element, (cr_val, d_val)) in self.dct_buffer.iter_mut().zip(group_c_rev_iter.zip(group_d_iter)) {
            *element = -cr_val - d_val;
        }

        //the second half of the dct input is is A - Br
        for (element, (a_val, br_val)) in self.dct_buffer[group_size..].iter_mut().zip(group_a_iter.zip(group_b_rev_iter)) {
            *element = a_val - br_val;
        }

        self.dct.process(&self.dct_buffer, spectrum);
    }




    /// Runs the IMDCT on the signal and puts int in the spectrum. we split the spectrum into two halves to make code elsewhere in this impl simpler
    /// The two halves are usually contiguous, but sometimes not
    fn process_inverse_internal(&mut self, signal: &[T], spectrum_left: &mut [T], spectrum_right: &mut [T]) {
        self.dct.process(signal, self.dct_buffer.as_mut_slice());

        let segment_size = self.dct_buffer.len();
        let group_size = segment_size / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in spectrum_left.iter_mut().zip(&self.window).zip(self.dct_buffer[group_size..].iter()) {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in spectrum_left.iter_mut().zip(&self.window).skip(group_size).zip(self.dct_buffer[group_size..].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in spectrum_right.iter_mut().zip(&self.window[segment_size..]).zip(self.dct_buffer[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in spectrum_right.iter_mut().zip(&self.window[segment_size..]).skip(group_size).zip(self.dct_buffer[..group_size].iter()) {
            *output = *output - *val * *window_val;
        }
    }

    /// Runs the IMDCT on the signal and puts int in the spectrum. this version is intended for ocomputing an overlapped inverse with only a single segment
    //in that case, the beginning segment and end segment are the same slice
    fn process_inverse_internal_single(&mut self, signal: &[T], spectrum: &mut [T]) {
        self.dct.process(signal, self.dct_buffer.as_mut_slice());

        let segment_size = self.dct_buffer.len();
        let group_size = segment_size / 2;

        //copy the second half of the DCT output into the result
        for ((output, window_val), val) in spectrum.iter_mut().zip(&self.window).zip(self.dct_buffer[group_size..].iter()) {
            *output = *output + *val * *window_val;
        }

        //copy the second half of the DCT output again, but this time reversed and negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(&self.window).skip(group_size).zip(self.dct_buffer[group_size..].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output into the result, reversde+negated
        for ((output, window_val), val) in spectrum.iter_mut().zip(&self.window[segment_size..]).zip(self.dct_buffer[..group_size].iter().rev()) {
            *output = *output - *val * *window_val;
        }

        //copy the first half of the DCT output again, but this time not reversed
        for ((output, window_val), val) in spectrum.iter_mut().zip(&self.window[segment_size..]).skip(group_size).zip(self.dct_buffer[..group_size].iter()) {
            *output = *output - *val * *window_val;
        }
    }
}

pub mod window_fn {
    use DCTnum;
    use num::{Float, FromPrimitive};
    use num::traits::FloatConst;

    pub fn mp3<T>(len: usize) -> Vec<T>
        where T: Float + FloatConst + DCTnum
    {
        let constant_term: T = T::PI() / FromPrimitive::from_usize(len).unwrap();
        let half: T = FromPrimitive::from_f32(0.5f32).unwrap();

        (0..len).map(|n| {
            let n_float: T = FromPrimitive::from_usize(n).unwrap();
            (constant_term * (n_float + half)).sin()
        }).collect()
    }

    pub fn vorbis<T>(len: usize) -> Vec<T>
        where T: Float + FloatConst + DCTnum
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
    fn test_fast_abc() {
        for i in 1..11 {
            let size = i * 4;
            let input = random_signal(size);


            //compute the forward direction
            let mut slow_output = vec![0f32; size / 2];
            execute_slow(&input, slow_output.as_mut_slice());

            let mut dct = MDCT::new(size / 2);
            let mut fast_output = vec![0f32; size / 2];
            dct.process(&input, fast_output.as_mut_slice());
            compare_float_vectors(&slow_output, &fast_output);

            //now compute the inverse
            let mut slow_inverse = vec![0f32; size];
            execute_slow_inverse(&slow_output, slow_inverse.as_mut_slice());

            let mut fast_inverse = vec![0f32; size];
            dct.process_inverse(&slow_output, fast_inverse.as_mut_slice());
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


    /// Verify that the 'overlapped' method and inverse produce the same output as naively computing the overlapped output
    #[test]
    fn test_overlapped() {

        let input : Vec<f32> = vec![1f32, 2f32, 3f32, 4f32, 6f32, 5f32, 6f32, -1f32];
        let segment_size = 2;

        let final_segment_input = vec![input[input.len()-2], input[input.len()-1], input[0], input[1]];

        let mut fast_overlapped_result = vec![0f32; input.len()];
        let mut fast_result = vec![0f32; input.len()];
        let mut slow_result = vec![0f32; input.len()];

        //run the dedicated process algorithm
        let mut dct = MDCT::new(segment_size);
        dct.process_overlapped(&input, fast_overlapped_result.as_mut_slice());

        //run the same operation, but with both the slow algorithm and our fast bare algorithm
        for i in 0..input.len()/segment_size - 1 {
            execute_slow(&input[segment_size*i..segment_size*(i+2)], &mut slow_result[segment_size*i..segment_size*(i+1)]);
            dct.process(&input[segment_size*i..segment_size*(i+2)], &mut fast_result[segment_size*i..segment_size*(i+1)]);
        }

        execute_slow(&final_segment_input, &mut slow_result[input.len() - segment_size..]);
        dct.process(&final_segment_input, &mut fast_result[input.len() - segment_size..]);

        compare_float_vectors(&fast_overlapped_result, &slow_result);
        compare_float_vectors(&fast_overlapped_result, &fast_result);

        //now verify that the inverse works as well
        let mut fast_overlapped_inverse = vec![0f32; input.len()];
        dct.process_inverse_overlapped(&slow_result, fast_overlapped_inverse.as_mut_slice());

        let mut fast_inverse = vec![0f32; input.len() + segment_size];
        let mut slow_inverse = vec![0f32; input.len() + segment_size];

        for i in 0..input.len()/segment_size {
            execute_slow_inverse(&slow_result[segment_size*i..segment_size*(i+1)], &mut slow_inverse[segment_size*i..segment_size*(i+2)]);
            dct.process_inverse(&slow_result[segment_size*i..segment_size*(i+1)], &mut fast_inverse[segment_size*i..segment_size*(i+2)]);
        }

        //part of the data for the first segment is sitting at the end of the array, add it to the beginning
        for i in 0..segment_size {
            fast_inverse[i] += fast_inverse[i+input.len()];
            slow_inverse[i] += slow_inverse[i+input.len()];
        }

        compare_float_vectors(&fast_overlapped_inverse, &slow_inverse[..input.len()]);
        compare_float_vectors(&fast_overlapped_inverse, &fast_inverse[..input.len()]);
    }

    /// Verify the TDAC property of the process_overalpped and process_inverse_overlapped functions
    #[test]
    fn test_tdac() {

        //for various sizes, we're going to verify that we can transform multiple consecutive overlapping segments
        //and then compute the inverse to get them back
        for num_segments in 4..5 {
            for i in 1..9 {
                let segment_size = i * 2;
                let signal = random_signal(segment_size * num_segments);

                let mut output_buffer = vec![0f32; signal.len()];
                let mut intermediate_buffer = vec![0f32; signal.len()];

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