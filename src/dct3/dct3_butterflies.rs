
use rustfft::num_complex::Complex;
use rustfft::Length;

use twiddles;
use dct3::DCT3;
use common;

pub struct DCT3Butterfly2 {}
impl DCT3Butterfly2 {
	pub fn new() -> Self {
		Self {}
	}
	pub unsafe fn process_inplace<T: common::DCTnum>(&self, buffer: &mut [T]) {
		let half_0 = *buffer.get_unchecked(0) * T::from_f32(0.5).unwrap();
		let frac_1 = *buffer.get_unchecked(1) * T::FRAC_1_SQRT_2();

        *buffer.get_unchecked_mut(0) = half_0 + frac_1;
        *buffer.get_unchecked_mut(1) = half_0 - frac_1;
	}
	unsafe fn process_direct<T: common::DCTnum>(buffer: &mut [T], zero: usize, one: usize) {
		let half_0 = *buffer.get_unchecked(zero) * T::from_f32(0.5).unwrap();
		let frac_1 = *buffer.get_unchecked(one) * T::FRAC_1_SQRT_2();

        *buffer.get_unchecked_mut(zero) = half_0 + frac_1;
        *buffer.get_unchecked_mut(one) = half_0 - frac_1;
	}
}
impl<T: common::DCTnum> DCT3<T> for DCT3Butterfly2 {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

		let half_0 = input[0] * T::from_f32(0.5).unwrap();
		let frac_1 = input[1] * T::FRAC_1_SQRT_2();

		output[0] = half_0 + frac_1;
		output[1] = half_0 - frac_1;  
    }
}
impl Length for DCT3Butterfly2 {
    fn len(&self) -> usize {
        2
    }
}

pub struct DCT3Butterfly4<T> {
	twiddle: Complex<T>,
}
impl<T: common::DCTnum> DCT3Butterfly4<T> {
	pub fn new() -> Self {
		Self {
			twiddle: twiddles::single_twiddle(1,16,true)
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT3SplitRadix with n = 4

		// inner DCT3 of size 2
		DCT3Butterfly2::process_direct(buffer, 0, 2);

		// inner DCT3 of size 1, then sclared by twiddle factors
		let lower_dct4 = *buffer.get_unchecked(1) * self.twiddle.re + *buffer.get_unchecked(3) * self.twiddle.im;
        let upper_dct4 = *buffer.get_unchecked(1) * self.twiddle.im - *buffer.get_unchecked(3) * self.twiddle.re;

		// Merge our results
		*buffer.get_unchecked_mut(1) = *buffer.get_unchecked(2) + upper_dct4;
		*buffer.get_unchecked_mut(3) = *buffer.get_unchecked(0) - lower_dct4;
		*buffer.get_unchecked_mut(0) = *buffer.get_unchecked(0) + lower_dct4;
		*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) - upper_dct4;
	}
}
impl<T: common::DCTnum> DCT3<T> for DCT3Butterfly4<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT3Butterfly4<T> {
    fn len(&self) -> usize {
        4
    }
}

pub struct DCT3Butterfly8<T> {
	butterfly4: DCT3Butterfly4<T>,
	butterfly2: DCT3Butterfly2,
	twiddles: [Complex<T>; 2],
}
impl<T: common::DCTnum> DCT3Butterfly8<T> {
	pub fn new() -> Self {
		Self {
			butterfly4: DCT3Butterfly4::new(),
			butterfly2: DCT3Butterfly2::new(),
			twiddles: [twiddles::single_twiddle(1,32,true), twiddles::single_twiddle(3,32,true)]
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT3SplitRadix with n = 8

		//process the evens
		let mut dct3_buffer = [
			*buffer.get_unchecked(0),
			*buffer.get_unchecked(2),
			*buffer.get_unchecked(4),
			*buffer.get_unchecked(6),
		];
		self.butterfly4.process_inplace(&mut dct3_buffer);

		//process the odds
		let mut recursive_buffer_n1 = [
			*buffer.get_unchecked(1) * T::from_usize(2).unwrap(),
			*buffer.get_unchecked(3) + *buffer.get_unchecked(5),
		];
		let mut recursive_buffer_n3 = [
			*buffer.get_unchecked(7) * T::from_usize(2).unwrap(),
			*buffer.get_unchecked(3) - *buffer.get_unchecked(5),
		];
		self.butterfly2.process_inplace(&mut recursive_buffer_n1);
		self.butterfly2.process_inplace(&mut recursive_buffer_n3);

		// flip the sign of the odd-indexed N3 results
		recursive_buffer_n3[1] = -recursive_buffer_n3[1];

		// merge the temp buffers into the final output
		for i in 0..2 {
			let twiddle = self.twiddles[i];

            let lower_dct4 = recursive_buffer_n1[i] * twiddle.re + recursive_buffer_n3[i] * twiddle.im;
            let upper_dct4 = recursive_buffer_n1[i] * twiddle.im - recursive_buffer_n3[i] * twiddle.re;

            let lower_dct3 = dct3_buffer[i];
            let upper_dct3 = dct3_buffer[3 - i];

            *buffer.get_unchecked_mut(i) =     lower_dct3 + lower_dct4;
            *buffer.get_unchecked_mut(7 - i) = lower_dct3 - lower_dct4;

            *buffer.get_unchecked_mut(3 - i) = upper_dct3 + upper_dct4;
            *buffer.get_unchecked_mut(4 + i) = upper_dct3 - upper_dct4;
		}
	}
}
impl<T: common::DCTnum> DCT3<T> for DCT3Butterfly8<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT3Butterfly8<T> {
    fn len(&self) -> usize {
        8
    }
}

pub struct DCT3Butterfly16<T> {
	butterfly8: DCT3Butterfly8<T>,
	butterfly4: DCT3Butterfly4<T>,
	twiddles: [Complex<T>; 4],
}
impl<T: common::DCTnum> DCT3Butterfly16<T> {
	pub fn new() -> Self {
		Self {
			butterfly8: DCT3Butterfly8::new(),
			butterfly4: DCT3Butterfly4::new(),
			twiddles: [ 
				twiddles::single_twiddle(1,64,true),
				twiddles::single_twiddle(3,64,true),
				twiddles::single_twiddle(5,64,true),
				twiddles::single_twiddle(7,64,true),
			],
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT3SplitRadix with n = 16

		//process the evens
		let mut dct3_buffer = [
			*buffer.get_unchecked(0),
			*buffer.get_unchecked(2),
			*buffer.get_unchecked(4),
			*buffer.get_unchecked(6),
			*buffer.get_unchecked(8),
			*buffer.get_unchecked(10),
			*buffer.get_unchecked(12),
			*buffer.get_unchecked(14)
		];
		self.butterfly8.process_inplace(&mut dct3_buffer);

		//process the odds
		let mut recursive_buffer_n1 = [
			*buffer.get_unchecked(1) * T::from_usize(2).unwrap(),
			*buffer.get_unchecked(3) + *buffer.get_unchecked(5),
			*buffer.get_unchecked(7) + *buffer.get_unchecked(9),
			*buffer.get_unchecked(11) + *buffer.get_unchecked(13),
		];
		let mut recursive_buffer_n3 = [
			*buffer.get_unchecked(15) * T::from_usize(2).unwrap(),
			*buffer.get_unchecked(11) - *buffer.get_unchecked(13),
			*buffer.get_unchecked(7) - *buffer.get_unchecked(9),
			*buffer.get_unchecked(3) - *buffer.get_unchecked(5),
		];
		self.butterfly4.process_inplace(&mut recursive_buffer_n1);
		self.butterfly4.process_inplace(&mut recursive_buffer_n3);

		// flip the sign of the odd-indexed N3 results
		recursive_buffer_n3[1] = -recursive_buffer_n3[1];
		recursive_buffer_n3[3] = -recursive_buffer_n3[3];

		// merge the temp buffers into the final output
		for i in 0..4 {
			let twiddle = self.twiddles[i];

            let lower_dct4 = recursive_buffer_n1[i] * twiddle.re + recursive_buffer_n3[i] * twiddle.im;
            let upper_dct4 = recursive_buffer_n1[i] * twiddle.im - recursive_buffer_n3[i] * twiddle.re;

            let lower_dct3 = dct3_buffer[i];
            let upper_dct3 = dct3_buffer[7 - i];

            *buffer.get_unchecked_mut(i) =     lower_dct3 + lower_dct4;
            *buffer.get_unchecked_mut(15 - i) = lower_dct3 - lower_dct4;

            *buffer.get_unchecked_mut(7 - i) = upper_dct3 + upper_dct4;
            *buffer.get_unchecked_mut(8 + i) = upper_dct3 - upper_dct4;
		}
	}
}
impl<T: common::DCTnum> DCT3<T> for DCT3Butterfly16<T> {
    fn process(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT3Butterfly16<T> {
    fn len(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dct3::DCT3Naive;
    use test_utils::{compare_float_vectors, random_signal};

     //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::new();
                let size = $size;
                println!("{}", size);

                let naive = DCT3Naive::new(size);

		        // set up buffers
		        let mut expected_input = random_signal(size);
		        let mut expected_output = vec![0f32; size];

		        let mut inplace_buffer = expected_input.clone();

		        let mut actual_input = expected_input.clone();
		        let mut actual_output = expected_output.clone();

		        // perform the test
		        naive.process(&mut expected_input, &mut expected_output);

		        unsafe { butterfly.process_inplace(&mut inplace_buffer); }

		        butterfly.process(&mut actual_input, &mut actual_output);
		        println!("");
		        println!("expected output: {:?}", expected_output);
		        println!("inplace output:  {:?}", inplace_buffer);
		        println!("process output:  {:?}", actual_output);

		        assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace() failed, length = {}", size);
		        assert!(compare_float_vectors(&expected_output, &actual_output), "process() failed, length = {}", size);
		    }
		)
	}
    test_butterfly_func!(test_dct3_butterfly2, DCT3Butterfly2, 2);
    test_butterfly_func!(test_dct3_butterfly4, DCT3Butterfly4, 4);
    test_butterfly_func!(test_dct3_butterfly8, DCT3Butterfly8, 8);
    test_butterfly_func!(test_dct3_butterfly16, DCT3Butterfly16, 16);
}