extern crate rustdct;
extern crate rand;

mod common;

use rustdct::mdct::window_fn;
use common::{test_dct1, test_dct2, test_dct3, test_dct4, test_mdct, test_imdct};

#[test]
fn test_dct1_accuracy() {
    test_dct1::compare_known_length(
    	&[1_f32, 1_f32],
    	&[1_f32, 0_f32]);
    test_dct1::compare_known_length(
    	&[1_f32, 2_f32, 3_f32, 5_f32],
    	&[8_f32, -2.5_f32, 0.5_f32, -1_f32]);
    test_dct1::compare_known_length(
    	&[1_f32, 2_f32, 3_f32, 5_f32, 1_f32, -3_f32],
    	&[10.0_f32, 2.1909830056250525_f32, -6.5450849718747373_f32, 3.3090169943749475_f32, -0.95491502812526274_f32, -1.0_f32]);

    for len in 2..20 {
        test_dct1::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
        test_dct1::planned_matches_naive(len);
    }
}


#[test]
fn test_dct2_accuracy() {
    test_dct2::compare_known_length(
    	&[1_f32, 1_f32],
    	&[2_f32, 0_f32]);
    test_dct2::compare_known_length(
    	&[1_f32, 1_f32, 1_f32, 1_f32],
    	&[4_f32, 0_f32, 0_f32, 0_f32]);
    test_dct2::compare_known_length(
    	&[4_f32, 1_f32, 6_f32, 2_f32, 8_f32],
    	&[21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32]);

    for len in 1..20 {
        test_dct2::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
        test_dct2::planned_matches_naive(len);
    }
}


#[test]
fn test_dct3_accuracy() {
    test_dct3::compare_known_length(
    	&[2_f32, 0_f32],
    	&[1_f32, 1_f32]);
    test_dct3::compare_known_length(
    	&[4_f32, 0_f32, 0_f32, 0_f32],
    	&[2_f32, 2_f32, 2_f32, 2_f32]);
    test_dct3::compare_known_length(
    	&[21_f32, -4.39201132_f32, 2.78115295_f32, -1.40008449_f32, 7.28115295_f32],
    	&[10_f32, 2.5_f32, 15_f32, 5_f32, 20_f32]);

    for len in 1..20 {
        test_dct3::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
        test_dct3::planned_matches_naive(len);
    }
}


#[test]
fn test_dct4_accuracy() {
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

    for len in 1..20 {
        test_dct4::planned_matches_naive(len);
    }
    for &len in &[50, 51, 100, 101] {
        test_dct4::planned_matches_naive(len);
    }
}

#[test]
fn test_mdct_accuracy() {
    for curent_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
        for len in 1..10 {
            test_mdct::planned_matches_naive(len*2, curent_window_fn);
        }
        for &len in &[50, 52] {
            test_mdct::planned_matches_naive(len*2, curent_window_fn);
        }
    }
}

#[test]
fn test_mdct_tdac() {
    for i in 1..10 {
        let len = i * 2;
        test_mdct::test_tdac(len, 1f32 / len as f32, window_fn::one);
    }
    for &i in &[50, 52] {
        let len = i * 2;
        test_mdct::test_tdac(len, 1f32 / len as f32, window_fn::one);
    }


    for curent_window_fn in &[window_fn::mp3, window_fn::vorbis] {
        for i in 1..10 {
            let len = i * 2;
            test_mdct::test_tdac(len, 2f32 / len as f32, curent_window_fn);
        }
        for &i in &[50, 52] {
            let len = i * 2;
            test_mdct::test_tdac(len, 2f32 / len as f32, curent_window_fn);
        }
    }
}


#[test]
fn test_imdct_accuracy() {
    for curent_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
        for len in 1..10 {
            test_imdct::planned_matches_naive(len*2, curent_window_fn);
        }
        for &len in &[50, 52] {
            test_imdct::planned_matches_naive(len*2, curent_window_fn);
        }
    }
}
