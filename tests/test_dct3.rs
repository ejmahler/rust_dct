extern crate rust_dct;
extern crate rand;

mod common;

use rust_dct::dct3::DCT3;
use common::{compare_float_vectors, random_signal};

fn compare_known_length(input: &[f32], known_output: &[f32]) {
	assert_eq!(input.len(), known_output.len(), "Input and known output must have the same length");

	let len = input.len();

	let mut naive_input = input.to_vec();
	let mut actual_input = input.to_vec();

	let mut naive_output = vec![0f32; len];
	let mut actual_output = vec![0f32; len];

	let mut naive_dct = rust_dct::dct3::DCT3Naive::new(len);

	let mut planner = rust_dct::DCTPlanner::new();
	let mut actual_dct = planner.plan_dct3(len);

	assert_eq!(actual_dct.len(), len, "dct3 planner created a DCT of incorrect length");


	naive_dct.process(&mut naive_input, &mut naive_output);
	actual_dct.process(&mut actual_input, &mut actual_output);

	assert!(compare_float_vectors(&known_output, &naive_output));
	assert!(compare_float_vectors(&known_output, &actual_output));
}

fn planned_matches_naive(len: usize) -> bool {
	let mut naive_input = random_signal(len);
	let mut actual_input = naive_input.clone();

	let mut naive_output = vec![0f32; len];
	let mut actual_output = vec![0f32; len];

	let mut naive_dct = rust_dct::dct3::DCT3Naive::new(len);

	let mut planner = rust_dct::DCTPlanner::new();
	let mut actual_dct = planner.plan_dct3(len);

	assert_eq!(actual_dct.len(), len, "dct3 planner created a DCT of incorrect length");


	naive_dct.process(&mut naive_input, &mut naive_output);
	actual_dct.process(&mut actual_input, &mut actual_output);

	compare_float_vectors(&naive_output, &actual_output)
}

#[test]
fn test_dct3_known_lengths() {
    compare_known_length(
    	&[2_f32, 0_f32],
    	&[1_f32, 1_f32]);
    compare_known_length(
    	&[4_f32, 0_f32, 0_f32, 0_f32],
    	&[2_f32, 2_f32, 2_f32, 2_f32]);
    compare_known_length(
    	&[21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
    	&[10_f32, 2.5_f32, 15_f32, 5_f32, 20_f32]);
}

#[test]
fn test_dct3_matches_naive() {
    for len in 1..20 {
    	assert!(planned_matches_naive(len), "i = {}", len);
    }
    for &len in &[50, 51, 100, 101] {
    	assert!(planned_matches_naive(len), "i = {}", len);
    }
}
