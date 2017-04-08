extern crate rust_dct;
extern crate rand;

mod common;

use common::{test_dct1, test_dct2, test_dct3, test_dct4};

#[test]
fn test_dct1_known_lengths() {
    test_dct1::compare_known_length(
    	&[1_f32, 1_f32],
    	&[1_f32, 0_f32]);
    test_dct1::compare_known_length(
    	&[1_f32, 2_f32, 3_f32, 5_f32],
    	&[8_f32, -2.5_f32, 0.5_f32, -1_f32]);
    test_dct1::compare_known_length(
    	&[1_f32, 2_f32, 3_f32, 5_f32, 1_f32, -3_f32],
    	&[10.0_f32, 2.1909830056250525_f32, -6.5450849718747373_f32, 3.3090169943749475_f32, -0.95491502812526274_f32, -1.0_f32]);
}

#[test]
fn test_dct1_matches_naive() {
    for len in 2..20 {
    	test_dct1::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
    	test_dct1::planned_matches_naive(len);
    }
}


#[test]
fn test_dct2_known_lengths() {
    test_dct2::compare_known_length(
    	&[1_f32, 1_f32],
    	&[2_f32, 0_f32]);
    test_dct2::compare_known_length(
    	&[1_f32, 1_f32, 1_f32, 1_f32],
    	&[4_f32, 0_f32, 0_f32, 0_f32]);
    test_dct2::compare_known_length(
    	&[4_f32, 1_f32, 6_f32, 2_f32, 8_f32],
    	&[21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32]);
}

#[test]
fn test_dct2_matches_naive() {
    for len in 1..20 {
    	test_dct2::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
    	test_dct2::planned_matches_naive(len);
    }
}


#[test]
fn test_dct3_known_lengths() {
    test_dct3::compare_known_length(
    	&[2_f32, 0_f32],
    	&[1_f32, 1_f32]);
    test_dct3::compare_known_length(
    	&[4_f32, 0_f32, 0_f32, 0_f32],
    	&[2_f32, 2_f32, 2_f32, 2_f32]);
    test_dct3::compare_known_length(
    	&[21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
    	&[10_f32, 2.5_f32, 15_f32, 5_f32, 20_f32]);
}

#[test]
fn test_dct3_matches_naive() {
    for len in 1..20 {
    	test_dct3::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
    	test_dct3::planned_matches_naive(len);
    }
}


#[test]
fn test_dct4_known_lengths() {
    test_dct4::compare_known_length(
    	&[0_f32,0_f32,0_f32,0_f32,0_f32],
    	&[0_f32,0_f32,0_f32,0_f32,0_f32]);
    test_dct4::compare_known_length(
    	&[1_f32,1_f32,1_f32,1_f32,1_f32],
    	&[3.19623_f32, -1.10134_f32, 0.707107_f32, -0.561163_f32, 0.506233_f32]);
    test_dct4::compare_known_length(
    	&[4.7015433_f32, -11.926178_f32, 27.098675_f32, -1.9793236_f32],
    	&[9.36402_f32, -19.242455_f32, 17.949997_f32, 32.01607_f32]);
    test_dct4::compare_known_length(
    	&[6_f32,9_f32,1_f32,5_f32,2_f32,6_f32,2_f32,-1_f32],
    	&[23.9103_f32, 0.201528_f32, 5.36073_f32, 2.53127_f32, -5.21319_f32, -0.240328_f32, -9.32464_f32, -5.56147_f32]);
}

#[test]
fn test_dct4_matches_naive() {
    for len in 1..20 {
    	test_dct4::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
    	test_dct4::planned_matches_naive(len);
    }
}
