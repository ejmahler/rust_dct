
use rustfft::num_complex::Complex;
use rustfft::Length;

use twiddles;
use ::{DCT2, DST2, DCT3, DST3, Type2and3};
use common;

#[allow(non_camel_case_types)]
pub struct Butterfly2_Type2and3 {}
impl Butterfly2_Type2and3 {
	pub fn new() -> Self {
		Self {}
	}
	pub unsafe fn process_inplace_dct2<T: common::DCTnum>(&self, buffer: &mut [T]) {
		let sum = *buffer.get_unchecked(0) + *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(1) = (*buffer.get_unchecked(0) - *buffer.get_unchecked(1)) * T::FRAC_1_SQRT_2();
        *buffer.get_unchecked_mut(0) = sum;
	}
	unsafe fn process_scattered_dct2<T: common::DCTnum>(buffer: &mut [T], zero: usize, one: usize) {
		let sum = *buffer.get_unchecked(zero) + *buffer.get_unchecked(one);
        *buffer.get_unchecked_mut(one) = (*buffer.get_unchecked(zero) - *buffer.get_unchecked(one)) * T::FRAC_1_SQRT_2();
        *buffer.get_unchecked_mut(zero) = sum;
	}

	pub unsafe fn process_inplace_dct3<T: common::DCTnum>(&self, buffer: &mut [T]) {
		let half_0 = *buffer.get_unchecked(0) * T::from_f32(0.5).unwrap();
		let frac_1 = *buffer.get_unchecked(1) * T::FRAC_1_SQRT_2();

        *buffer.get_unchecked_mut(0) = half_0 + frac_1;
        *buffer.get_unchecked_mut(1) = half_0 - frac_1;
	}
	unsafe fn process_scattered_dct3<T: common::DCTnum>(buffer: &mut [T], zero: usize, one: usize) {
		let half_0 = *buffer.get_unchecked(zero) * T::from_f32(0.5).unwrap();
		let frac_1 = *buffer.get_unchecked(one) * T::FRAC_1_SQRT_2();

        *buffer.get_unchecked_mut(zero) = half_0 + frac_1;
        *buffer.get_unchecked_mut(one) = half_0 - frac_1;
	}

    pub unsafe fn process_inplace_dst2<T: common::DCTnum>(&self, buffer: &mut [T]) {
		let sum = *buffer.get_unchecked(0) - *buffer.get_unchecked(1);
        *buffer.get_unchecked_mut(0) = (*buffer.get_unchecked(0) + *buffer.get_unchecked(1)) * T::FRAC_1_SQRT_2();
        *buffer.get_unchecked_mut(1) = sum;
	}

    pub unsafe fn process_inplace_dst3<T: common::DCTnum>(&self, buffer: &mut [T]) {
		
		let frac_0 = *buffer.get_unchecked(0) * T::FRAC_1_SQRT_2();
        let half_1 = *buffer.get_unchecked(1) * T::from_f32(0.5).unwrap();

        *buffer.get_unchecked_mut(0) = frac_0 + half_1;
        *buffer.get_unchecked_mut(1) = frac_0 - half_1;
	}
}
impl<T: common::DCTnum> DCT2<T> for Butterfly2_Type2and3 {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        output[0] = input[0] + input[1];
        output[1] = (input[0] - input[1]) * T::FRAC_1_SQRT_2();      
    }
}
impl<T: common::DCTnum> DCT3<T> for Butterfly2_Type2and3 {
    fn process_dct3(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

		let half_0 = input[0] * T::from_f32(0.5).unwrap();
		let frac_1 = input[1] * T::FRAC_1_SQRT_2();

		output[0] = half_0 + frac_1;
		output[1] = half_0 - frac_1;  
    }
}
impl<T: common::DCTnum> DST2<T> for Butterfly2_Type2and3 {
    fn process_dst2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        output[0] = (input[0] + input[1]) * T::FRAC_1_SQRT_2();
        output[1] = input[0] - input[1];      
    }
}
impl<T: common::DCTnum> DST3<T> for Butterfly2_Type2and3 {
    fn process_dst3(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());

        let frac_0 = input[0] * T::FRAC_1_SQRT_2();
		let half_1 = input[1] * T::from_f32(0.5).unwrap();

		output[0] = frac_0 + half_1;  
        output[1] = frac_0 - half_1;
    }
}
impl<T: common::DCTnum> Type2and3<T> for Butterfly2_Type2and3{}
impl Length for Butterfly2_Type2and3 {
    fn len(&self) -> usize {
        2
    }
}

#[allow(non_camel_case_types)]
pub struct Butterfly4_Type2and3<T> {
	twiddle: Complex<T>,
}
impl<T: common::DCTnum> Butterfly4_Type2and3<T> {
	pub fn new() -> Self {
		Self {
			twiddle: twiddles::single_twiddle(1,16).conj()
		}
	}
	pub unsafe fn process_inplace_dct2(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT2SplitRadix with n = 4

		let lower_dct4 = *buffer.get_unchecked(0) - *buffer.get_unchecked(3);
		let upper_dct4 = *buffer.get_unchecked(2) - *buffer.get_unchecked(1);

		*buffer.get_unchecked_mut(0) = *buffer.get_unchecked(0) + *buffer.get_unchecked(3);
		*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) + *buffer.get_unchecked(1);

        Butterfly2_Type2and3::process_scattered_dct2(buffer, 0, 2);

        *buffer.get_unchecked_mut(1) = lower_dct4 * self.twiddle.re - upper_dct4 * self.twiddle.im;
        *buffer.get_unchecked_mut(3) = upper_dct4 * self.twiddle.re + lower_dct4 * self.twiddle.im;
	}
    pub unsafe fn process_inplace_dct3(&self, buffer: &mut [T]) {
		// perform a step of split radix -- derived from DCT3SplitRadix with n = 4

		// inner DCT3 of size 2
		Butterfly2_Type2and3::process_scattered_dct3(buffer, 0, 2);

		// inner DCT3 of size 1, then sclared by twiddle factors
		let lower_dct4 = *buffer.get_unchecked(1) * self.twiddle.re + *buffer.get_unchecked(3) * self.twiddle.im;
        let upper_dct4 = *buffer.get_unchecked(1) * self.twiddle.im - *buffer.get_unchecked(3) * self.twiddle.re;

		// Merge our results
		*buffer.get_unchecked_mut(1) = *buffer.get_unchecked(2) + upper_dct4;
		*buffer.get_unchecked_mut(3) = *buffer.get_unchecked(0) - lower_dct4;
		*buffer.get_unchecked_mut(0) = *buffer.get_unchecked(0) + lower_dct4;
		*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(2) - upper_dct4;
	}
    pub unsafe fn process_inplace_dst2(&self, buffer: &mut [T]) {
		// Derived from process_inplace_dct2 by negating the odd inputs, and reversing the outputs

		let lower_dct4 = *buffer.get_unchecked(0) + *buffer.get_unchecked(3);
		let upper_dct4 = *buffer.get_unchecked(2) + *buffer.get_unchecked(1);

		*buffer.get_unchecked_mut(3) = *buffer.get_unchecked(0) - *buffer.get_unchecked(3);
		*buffer.get_unchecked_mut(1) = *buffer.get_unchecked(2) - *buffer.get_unchecked(1);

        Butterfly2_Type2and3::process_scattered_dct2(buffer, 3, 1);

        *buffer.get_unchecked_mut(2) = lower_dct4 * self.twiddle.re - upper_dct4 * self.twiddle.im;
        *buffer.get_unchecked_mut(0) = upper_dct4 * self.twiddle.re + lower_dct4 * self.twiddle.im;
	}
    pub unsafe fn process_inplace_dst3(&self, buffer: &mut [T]) {
		// Derived from process_inplace_dst3 by reversing the inputs, and negating the odd outputs

		// inner DCT3 of size 2
		Butterfly2_Type2and3::process_scattered_dct3(buffer, 3, 1);

		// inner DCT3 of size 1, then sclared by twiddle factors
		let lower_dct4 = *buffer.get_unchecked(2) * self.twiddle.re + *buffer.get_unchecked(0) * self.twiddle.im;
        let upper_dct4 = *buffer.get_unchecked(2) * self.twiddle.im - *buffer.get_unchecked(0) * self.twiddle.re;

		// Merge our results
		*buffer.get_unchecked_mut(0) = *buffer.get_unchecked(3) + lower_dct4;
		*buffer.get_unchecked_mut(2) = *buffer.get_unchecked(1) - upper_dct4;
        *buffer.get_unchecked_mut(1) = -(*buffer.get_unchecked(1) + upper_dct4);
		*buffer.get_unchecked_mut(3) = lower_dct4 - *buffer.get_unchecked(3);
	}
}
impl<T: common::DCTnum> DCT2<T> for Butterfly4_Type2and3<T> {
    fn process_dct2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace_dct2(output); }
    }
}
impl<T: common::DCTnum> DCT3<T> for Butterfly4_Type2and3<T> {
    fn process_dct3(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace_dct3(output); }
    }
}
impl<T: common::DCTnum> DST2<T> for Butterfly4_Type2and3<T> {
    fn process_dst2(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace_dst2(output); }
    }
}
impl<T: common::DCTnum> DST3<T> for Butterfly4_Type2and3<T> {
    fn process_dst3(&self, input: &mut [T], output: &mut [T]) {
        common::verify_length(input, output, self.len());
		
        output.copy_from_slice(input);
        unsafe { self.process_inplace_dst3(output); }
    }
}
impl<T: common::DCTnum> Type2and3<T> for Butterfly4_Type2and3<T>{}
impl<T> Length for Butterfly4_Type2and3<T> {
    fn len(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod test {
    use super::*;

    //the tests for all butterflies will be identical except for the identifiers used and size
    //so it's ideal for a macro
    macro_rules! test_butterfly_func {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            mod $test_name {
                use super::*;
                use algorithm::NaiveType2And3;
                use test_utils::{compare_float_vectors, random_signal};
                #[test]
                fn $test_name() {
                    let size = $size;
                    println!("{}", size);

                    let naive = NaiveType2And3::new(size);

                    check_dct2(&naive);
                    check_dct3(&naive);
                    check_dst2(&naive);
                    check_dst3(&naive);
                }

                fn check_dct2(naive_instance: &NaiveType2And3<f32>) {
                    let butterfly_instance = $struct_name::new();

                    // set up buffers
                    let expected_input = random_signal($size);
                    
                    let mut expected_output = vec![0f32; $size];
                    let mut inplace_buffer = expected_input.clone();
                    let mut actual_output = expected_output.clone();

                    // perform the test
                    naive_instance.process_dct2(&mut expected_input.clone(), &mut expected_output);

                    unsafe { butterfly_instance.process_inplace_dct2(&mut inplace_buffer); }

                    butterfly_instance.process_dct2(&mut expected_input.clone(), &mut actual_output);
                    println!("");
                    println!("expected output: {:?}", expected_output);
                    println!("inplace output:  {:?}", inplace_buffer);
                    println!("process output:  {:?}", actual_output);

                    assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace_dct2() failed, length = {}", $size);
                    assert!(compare_float_vectors(&expected_output, &actual_output), "process_dct2() failed, length = {}", $size);
                }

                fn check_dct3(naive_instance: &NaiveType2And3<f32>) {
                    let butterfly_instance = $struct_name::new();

                    // set up buffers
                    let expected_input = random_signal($size);
                    
                    let mut expected_output = vec![0f32; $size];
                    let mut inplace_buffer = expected_input.clone();
                    let mut actual_output = expected_output.clone();

                    // perform the test
                    naive_instance.process_dct3(&mut expected_input.clone(), &mut expected_output);

                    unsafe { butterfly_instance.process_inplace_dct3(&mut inplace_buffer); }

                    butterfly_instance.process_dct3(&mut expected_input.clone(), &mut actual_output);
                    println!("");
                    println!("expected output: {:?}", expected_output);
                    println!("inplace output:  {:?}", inplace_buffer);
                    println!("process output:  {:?}", actual_output);

                    assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace_dct3() failed, length = {}", $size);
                    assert!(compare_float_vectors(&expected_output, &actual_output), "process_dct3() failed, length = {}", $size);
                }

                fn check_dst2(naive_instance: &NaiveType2And3<f32>) {
                    let butterfly_instance = $struct_name::new();

                    // set up buffers
                    let expected_input = random_signal($size);
                    
                    let mut expected_output = vec![0f32; $size];
                    let mut inplace_buffer = expected_input.clone();
                    let mut actual_output = expected_output.clone();

                    // perform the test
                    naive_instance.process_dst2(&mut expected_input.clone(), &mut expected_output);

                    unsafe { butterfly_instance.process_inplace_dst2(&mut inplace_buffer); }

                    butterfly_instance.process_dst2(&mut expected_input.clone(), &mut actual_output);
                    println!("");
                    println!("expected output: {:?}", expected_output);
                    println!("inplace output:  {:?}", inplace_buffer);
                    println!("process output:  {:?}", actual_output);

                    assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace_dst2() failed, length = {}", $size);
                    assert!(compare_float_vectors(&expected_output, &actual_output), "process_dst2() failed, length = {}", $size);
                }

                fn check_dst3(naive_instance: &NaiveType2And3<f32>) {
                    let butterfly_instance = $struct_name::new();

                    // set up buffers
                    let expected_input = random_signal($size);
                    
                    let mut expected_output = vec![0f32; $size];
                    let mut inplace_buffer = expected_input.clone();
                    let mut actual_output = expected_output.clone();

                    // perform the test
                    naive_instance.process_dst3(&mut expected_input.clone(), &mut expected_output);

                    unsafe { butterfly_instance.process_inplace_dst3(&mut inplace_buffer); }

                    butterfly_instance.process_dst3(&mut expected_input.clone(), &mut actual_output);
                    println!("");
                    println!("expected output: {:?}", expected_output);
                    println!("inplace output:  {:?}", inplace_buffer);
                    println!("process output:  {:?}", actual_output);

                    assert!(compare_float_vectors(&expected_output, &inplace_buffer), "process_inplace_dst3() failed, length = {}", $size);
                    assert!(compare_float_vectors(&expected_output, &actual_output), "process_dst3() failed, length = {}", $size);
                }
            }
        )
    }
    test_butterfly_func!(test_butterfly2_type2and3, Butterfly2_Type2and3, 2);
    test_butterfly_func!(test_butterfly4_type2and3, Butterfly4_Type2and3, 4);
}