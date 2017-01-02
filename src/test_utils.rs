use super::*;
use std::f32;

use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};

pub fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
    a >= b - tolerance && a <= b + tolerance
}

pub fn compare_float_vectors(expected: &[f32], observed: &[f32]) {
    assert_eq!(expected.len(), observed.len());

    let tolerance: f32 = 0.001;

    for i in 0..expected.len() {
        assert!(fuzzy_cmp(observed[i], expected[i], tolerance));
    }
}

pub fn random_signal(length: usize) -> Vec<f32> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);

    let seed : [usize; 5] = [1910, 11431, 4984, 14828, length];
    let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

    for _ in 0..length {
        sig.push(normal_dist.ind_sample(&mut rng) as f32);
    }
    return sig;
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

        compare_float_vectors(&input.as_slice(), &output.as_slice());
    }
}