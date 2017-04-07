extern crate rust_dct;
extern crate rand;

mod common;

use rust_dct::dct4::DCT4;
use common::{compare_float_vectors, random_signal};

fn compare_known_length(input: &[f32], known_output: &[f32]) {
	assert_eq!(input.len(), known_output.len(), "Input and known output must have the same length");

	let len = input.len();

	let mut naive_input = input.to_vec();
	let mut actual_input = input.to_vec();

	let mut naive_output = vec![0f32; len];
	let mut actual_output = vec![0f32; len];

	let mut naive_dct = rust_dct::dct4::DCT4Naive::new(len);

	let mut planner = rust_dct::DCTPlanner::new();
	let mut actual_dct = planner.plan_dct4(len);

	assert_eq!(actual_dct.len(), len, "dct4 planner created a DCT of incorrect length");


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

	let mut naive_dct = rust_dct::dct4::DCT4Naive::new(len);

	let mut planner = rust_dct::DCTPlanner::new();
	let mut actual_dct = planner.plan_dct4(len);

	assert_eq!(actual_dct.len(), len, "dct4 planner created a DCT of incorrect length");


	naive_dct.process(&mut naive_input, &mut naive_output);
	actual_dct.process(&mut actual_input, &mut actual_output);

	compare_float_vectors(&naive_output, &actual_output)
}

#[test]
fn test_dct4_known_lengths() {
    compare_known_length(
    	&[0_f32,0_f32,0_f32,0_f32,0_f32],
    	&[0_f32,0_f32,0_f32,0_f32,0_f32]);
    compare_known_length(
    	&[1_f32,1_f32,1_f32,1_f32,1_f32],
    	&[3.19623_f32, -1.10134_f32, 0.707107_f32, -0.561163_f32, 0.506233_f32]);
    compare_known_length(
    	&[4.7015433_f32, -11.926178_f32, 27.098675_f32, -1.9793236_f32],
    	&[9.36402_f32, -19.242455_f32, 17.949997_f32, 32.01607_f32]);
    compare_known_length(
    	&[6_f32,9_f32,1_f32,5_f32,2_f32,6_f32,2_f32,-1_f32],
    	&[23.9103_f32, 0.201528_f32, 5.36073_f32, 2.53127_f32, -5.21319_f32, -0.240328_f32, -9.32464_f32, -5.56147_f32]);
}

#[test]
fn test_dct4_matches_naive() {
    for len in 1..20 {
    	assert!(planned_matches_naive(len), "i = {}", len);
    }
    for &len in &[50, 51, 201, 202] {
    	assert!(planned_matches_naive(len), "i = {}", len);
    }
}
