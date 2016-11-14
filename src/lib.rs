
extern crate num;
extern crate rustfft;

mod dct_type_2;
mod dct_type_3;
mod dct_type_4;

mod math_utils;

pub use self::dct_type_2::DCT2;
pub use self::dct_type_3::DCT3;
pub use self::dct_type_4::DCT4;

pub use self::dct_type_2::dct2_2d;
pub use self::dct_type_3::dct3_2d;


#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
        a >= b - tolerance && a <= b + tolerance
    }

    fn compare_float_vectors(expected: &[f32], observed: &[f32]) {
        assert_eq!(expected.len(), observed.len());

        let tolerance: f32 = 0.0001;

        for i in 0..expected.len() {
            assert!(fuzzy_cmp(observed[i], expected[i], tolerance));
        }
    }

    #[test]
    fn test_dct2_dct3_inverse() {

        let input_list = vec![
    		vec![1_f32, 1_f32],
    		vec![1_f32, 1_f32, 1_f32, 1_f32, 1_f32],
    		vec![1_f32, 2_f32],
    		vec![1_f32, 9_f32, 1_f32, 2_f32, 3_f32],
    	];

        for input in input_list {
            let mut midpoint = input.clone();
            let mut output = input.clone();

            let mut dct2 = DCT2::new(input.len());
            dct2.process(input.as_slice(), midpoint.as_mut_slice());

            let mut dct3 = DCT3::new(input.len());
            dct3.process(midpoint.as_slice(), output.as_mut_slice());

            // scale the result by 2/N
            let scale = 2_f32 / input.len() as f32;
            for item in output.iter_mut() {
                *item *= scale
            }

            println!("");
            println!("{:?}", input);
            println!("{:?}", midpoint);
            println!("{:?}", output);

            compare_float_vectors(&input.as_slice(), &output.as_slice());
        }
    }

    #[test]
    fn test_2d_dct2_dct3_inverse() {

        let input_list = vec![
    		(2 as usize, 2 as usize, vec![
    			1_f32, 1_f32,
    			1_f32, 1_f32,
    		]),
    		(3 as usize, 2 as usize, vec![
    			1_f32, 1_f32, 1_f32,
    			1_f32, 1_f32, 1_f32
    		]),
    		(2 as usize, 3 as usize, vec![
    			1_f32, 2_f32,
    			1_f32, 2_f32,
    			2_f32, 3_f32,

    		]),
    	];

        for (width, height, input) in input_list {
            let mut midpoint = input.clone();
            dct2_2d(width, height, &mut midpoint);

            let mut output = midpoint.clone();
            dct3_2d(width, height, &mut output);

            // scale the result by 4/N
            let scale = 4_f32 / input.len() as f32;
            for item in output.iter_mut() {
                *item *= scale
            }

            println!("");
            println!("{:?}", input);
            println!("{:?}", midpoint);
            println!("{:?}", output);

            compare_float_vectors(&input.as_slice(), &output.as_slice());
        }
    }
}
