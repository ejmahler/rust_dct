
use rustfft::num_complex::Complex;
use rustfft::Length;

use twiddles;
use ::DCT2;
use common;

pub struct DCT2Butterfly2 {}
impl DCT2Butterfly2 {
	pub fn new() -> Self {
		Self {}
	}
	pub unsafe fn process_inplace<T: common::DCTnum>(&self, buffer: &mut [T]) {
		let sum = *buffer.get_unchecked(0) + *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(1) = (*buffer.get_unchecked(0) - *buffer.get_unchecked(1)) * T::FRAC_1_SQRT_2();
        *buffer.get_unchecked_mut(0) = sum;
	}
	unsafe fn process_direct<T: common::DCTnum>(buffer: &mut [T], zero: usize, one: usize) {
		let sum = *buffer.get_unchecked(zero) + *buffer.get_unchecked(one);
        *buffer.get_unchecked_mut(one) = (*buffer.get_unchecked(zero) - *buffer.get_unchecked(one)) * T::FRAC_1_SQRT_2();
        *buffer.get_unchecked_mut(zero) = sum;
	}
}
impl<T: common::DCTnum> DCT2<T> for DCT2Butterfly2 {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

		output[0] = input[0] + input[1];
        output[1] = (input[0] - input[1]) * T::FRAC_1_SQRT_2();      
    }
}
impl Length for DCT2Butterfly2 {
    fn len(&self) -> usize {
        2
    }
}

pub struct DCT2Butterfly4<T> {
	twiddle: Complex<T>,
}
impl<T: common::DCTnum> DCT2Butterfly4<T> {
	pub fn new() -> Self {
		Self {
			twiddle: twiddles::single_twiddle(1,16).conj()
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT2SplitRadix with n = 4

		let lower_dct4 = *buffer.get_unchecked(0) - *buffer.get_unchecked(3);
		let upper_dct4 = *buffer.get_unchecked(2) - *buffer.get_unchecked(1);

		*buffer.get_unchecked_mut(0) = *buffer.get_unchecked(0) + *buffer.get_unchecked(3);
		*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) + *buffer.get_unchecked(1);

        DCT2Butterfly2::process_direct(buffer, 0, 2);

        *buffer.get_unchecked_mut(1) = lower_dct4 * self.twiddle.re - upper_dct4 * self.twiddle.im;
        *buffer.get_unchecked_mut(3) = upper_dct4 * self.twiddle.re + lower_dct4 * self.twiddle.im;
	}
}
impl<T: common::DCTnum> DCT2<T> for DCT2Butterfly4<T> {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT2Butterfly4<T> {
    fn len(&self) -> usize {
        4
    }
}

pub struct DCT2Butterfly8<T> {
	butterfly4: DCT2Butterfly4<T>,
	butterfly2: DCT2Butterfly2,
	twiddle0: Complex<T>,
	twiddle1: Complex<T>,
}
impl<T: common::DCTnum> DCT2Butterfly8<T> {
	pub fn new() -> Self {
		Self {
			butterfly4: DCT2Butterfly4::new(),
			butterfly2: DCT2Butterfly2::new(),
			twiddle0: twiddles::single_twiddle(1,32).conj(),
			twiddle1: twiddles::single_twiddle(3,32).conj(),
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT2SplitRadix with n = 8

		//process the evens
		let mut dct2_buffer = [
			*buffer.get_unchecked(0) + *buffer.get_unchecked(7),
			*buffer.get_unchecked(1) + *buffer.get_unchecked(6),
			*buffer.get_unchecked(2) + *buffer.get_unchecked(5),
			*buffer.get_unchecked(3) + *buffer.get_unchecked(4),
		];
		self.butterfly4.process_inplace(&mut dct2_buffer);

		//process the odds
		let differences = [
			*buffer.get_unchecked(0) - *buffer.get_unchecked(7),
			*buffer.get_unchecked(3) - *buffer.get_unchecked(4),
			*buffer.get_unchecked(1) - *buffer.get_unchecked(6),
			*buffer.get_unchecked(2) - *buffer.get_unchecked(5),
		];

		let mut dct4_even_buffer = [
			differences[0] * self.twiddle0.re + differences[1] * self.twiddle0.im,
			differences[2] * self.twiddle1.re + differences[3] * self.twiddle1.im,
		];
		let mut dct4_odd_buffer = [
			differences[3] * self.twiddle1.re - differences[2] * self.twiddle1.im,
			differences[0] * self.twiddle0.im - differences[1] * self.twiddle0.re,
		];

		self.butterfly2.process_inplace(&mut dct4_even_buffer);
		self.butterfly2.process_inplace(&mut dct4_odd_buffer);

		// combine the results
		*buffer.get_unchecked_mut(0) = dct2_buffer[0];
		*buffer.get_unchecked_mut(1) = dct4_even_buffer[0];
		*buffer.get_unchecked_mut(2) = dct2_buffer[1];
		*buffer.get_unchecked_mut(3) = dct4_even_buffer[1] - dct4_odd_buffer[1];
		*buffer.get_unchecked_mut(4) = dct2_buffer[2];
		*buffer.get_unchecked_mut(5) = dct4_even_buffer[1] + dct4_odd_buffer[1];
		*buffer.get_unchecked_mut(6) = dct2_buffer[3];
		*buffer.get_unchecked_mut(7) = dct4_odd_buffer[0];

	}
}
impl<T: common::DCTnum> DCT2<T> for DCT2Butterfly8<T> {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT2Butterfly8<T> {
    fn len(&self) -> usize {
        8
    }
}

pub struct DCT2Butterfly16<T> {
	butterfly8: DCT2Butterfly8<T>,
	butterfly4: DCT2Butterfly4<T>,
	twiddle0: Complex<T>,
	twiddle1: Complex<T>,
	twiddle2: Complex<T>,
	twiddle3: Complex<T>,
}
impl<T: common::DCTnum> DCT2Butterfly16<T> {
	pub fn new() -> Self {
		Self {
			butterfly8: DCT2Butterfly8::new(),
			butterfly4: DCT2Butterfly4::new(),
			twiddle0: twiddles::single_twiddle(1,64).conj(),
			twiddle1: twiddles::single_twiddle(3,64).conj(),
			twiddle2: twiddles::single_twiddle(5,64).conj(),
			twiddle3: twiddles::single_twiddle(7,64).conj(),
		}
	}
	pub unsafe fn process_inplace(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT2SplitRadix with n = 16

		//process the evens
		let mut dct2_buffer = [
			*buffer.get_unchecked(0) + *buffer.get_unchecked(15),
			*buffer.get_unchecked(1) + *buffer.get_unchecked(14),
			*buffer.get_unchecked(2) + *buffer.get_unchecked(13),
			*buffer.get_unchecked(3) + *buffer.get_unchecked(12),
			*buffer.get_unchecked(4) + *buffer.get_unchecked(11),
			*buffer.get_unchecked(5) + *buffer.get_unchecked(10),
			*buffer.get_unchecked(6) + *buffer.get_unchecked(9),
			*buffer.get_unchecked(7) + *buffer.get_unchecked(8),
		];
		self.butterfly8.process_inplace(&mut dct2_buffer);

		//process the odds
		let differences = [
			*buffer.get_unchecked(0) - *buffer.get_unchecked(15),
			*buffer.get_unchecked(7) - *buffer.get_unchecked(8),
			*buffer.get_unchecked(1) - *buffer.get_unchecked(14),
			*buffer.get_unchecked(6) - *buffer.get_unchecked(9),
			*buffer.get_unchecked(2) - *buffer.get_unchecked(13),
			*buffer.get_unchecked(5) - *buffer.get_unchecked(10),
			*buffer.get_unchecked(3) - *buffer.get_unchecked(12),
			*buffer.get_unchecked(4) - *buffer.get_unchecked(11),
		];

		let mut dct4_even_buffer = [
			differences[0] * self.twiddle0.re + differences[1] * self.twiddle0.im,
			differences[2] * self.twiddle1.re + differences[3] * self.twiddle1.im,
			differences[4] * self.twiddle2.re + differences[5] * self.twiddle2.im,
			differences[6] * self.twiddle3.re + differences[7] * self.twiddle3.im,
		];
		let mut dct4_odd_buffer = [
			differences[7] * self.twiddle3.re - differences[6] * self.twiddle3.im,
			differences[4] * self.twiddle2.im - differences[5] * self.twiddle2.re,
			differences[3] * self.twiddle1.re - differences[2] * self.twiddle1.im,
			differences[0] * self.twiddle0.im - differences[1] * self.twiddle0.re,
		];

		self.butterfly4.process_inplace(&mut dct4_even_buffer);
		self.butterfly4.process_inplace(&mut dct4_odd_buffer);

		// combine the results
		*buffer.get_unchecked_mut(0) =  dct2_buffer[0];
		*buffer.get_unchecked_mut(1) =  dct4_even_buffer[0];
		*buffer.get_unchecked_mut(2) =  dct2_buffer[1];
		*buffer.get_unchecked_mut(3) =  dct4_even_buffer[1] - dct4_odd_buffer[3];
		*buffer.get_unchecked_mut(4) =  dct2_buffer[2];
		*buffer.get_unchecked_mut(5) =  dct4_even_buffer[1] + dct4_odd_buffer[3];
		*buffer.get_unchecked_mut(6) =  dct2_buffer[3];
		*buffer.get_unchecked_mut(7) =  dct4_even_buffer[2] + dct4_odd_buffer[2];
		*buffer.get_unchecked_mut(8) =  dct2_buffer[4];
		*buffer.get_unchecked_mut(9) =  dct4_even_buffer[2] - dct4_odd_buffer[2];
		*buffer.get_unchecked_mut(10) = dct2_buffer[5];
		*buffer.get_unchecked_mut(11) = dct4_even_buffer[3] - dct4_odd_buffer[1];
		*buffer.get_unchecked_mut(12) = dct2_buffer[6];
		*buffer.get_unchecked_mut(13) = dct4_even_buffer[3] + dct4_odd_buffer[1];
		*buffer.get_unchecked_mut(14) = dct2_buffer[7];
		*buffer.get_unchecked_mut(15) = dct4_odd_buffer[0];

	}
}
impl<T: common::DCTnum> DCT2<T> for DCT2Butterfly16<T> {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace(output); }
    }
}
impl<T> Length for DCT2Butterfly16<T> {
    fn len(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod test {
    use super::*;
	use algorithm::NaiveType23;
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

                let naive = NaiveType23::new(size);

		        // set up buffers
		        let mut expected_input = random_signal(size);
		        let mut expected_output = vec![0f32; size];

		        let mut inplace_buffer = expected_input.clone();

		        let mut actual_input = expected_input.clone();
		        let mut actual_output = expected_output.clone();

		        // perform the test
		        naive.process_dct2(&mut expected_input, &mut expected_output);

		        unsafe { butterfly.process_inplace(&mut inplace_buffer); }

		        butterfly.process_dct2(&mut actual_input, &mut actual_output);
		        println!("");
		        println!("expected output: {:?}", expected_output);
		        println!("inplace output:  {:?}", inplace_buffer);
		        println!("process output:  {:?}", actual_output);

		        assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace() failed, length = {}", size);
		        assert!(compare_float_vectors(&expected_output, &actual_output), "process() failed, length = {}", size);
		    }
		)
	}
    test_butterfly_func!(test_dct2_butterfly2, DCT2Butterfly2, 2);
    test_butterfly_func!(test_dct2_butterfly4, DCT2Butterfly4, 4);
    test_butterfly_func!(test_dct2_butterfly8, DCT2Butterfly8, 8);
    test_butterfly_func!(test_dct2_butterfly16, DCT2Butterfly16, 16);
}