extern crate rustdct;
extern crate rand;

#[macro_use]
mod common;

use rustdct::dct1::{DCT1, DCT1Naive};
use rustdct::dct2::{DCT2, DCT2Naive};
use rustdct::dct3::{DCT3, DCT3Naive};
use rustdct::dct4::{DCT4, DCT4Naive};
use rustdct::dst::{DST1, DST2, DST3, DST4, DST1Naive, DST2Naive, DST3Naive, DST4Naive};
use rustdct::mdct::window_fn;
use rustdct::DCTplanner;

use common::macros::{test_mdct, test_imdct};
use common::known_data::*;
use common::slow_fns::*;
use common::{random_signal, compare_float_vectors};

#[test]
fn test_dct1_accuracy() {
    dct_test_with_known_data!(DCT1Naive, process_dct1, slow_dct1, known_values_dct1);
    dct_test_with_planner!(DCT1Naive, process_dct1, plan_dct1, 2);
}

#[test]
fn test_dct2_accuracy() {
    dct_test_with_known_data!(DCT2Naive, process_dct2, slow_dct2, known_values_dct2);
    dct_test_with_planner!(DCT2Naive, process_dct2, plan_dct2, 1);
}

#[test]
fn test_dct3_accuracy() {
    dct_test_with_known_data!(DCT3Naive, process_dct3, slow_dct3, known_values_dct3);
    dct_test_with_planner!(DCT3Naive, process_dct3, plan_dct3, 1);
}

#[test]
fn test_dct4_accuracy() {
    dct_test_with_known_data!(DCT4Naive, process_dct4, slow_dct4, known_values_dct4);
    dct_test_with_planner!(DCT4Naive, process_dct4, plan_dct4, 1);
}

#[test]
fn test_dst1_accuracy() {
    dct_test_with_known_data!(DST1Naive, process_dst1, slow_dst1, known_values_dst1);
}

#[test]
fn test_dst2_accuracy() {
    dct_test_with_known_data!(DST2Naive, process_dst2, slow_dst2, known_values_dst2);
}

#[test]
fn test_dst3_accuracy() {
    dct_test_with_known_data!(DST3Naive, process_dst3, slow_dst3, known_values_dst3);
}

#[test]
fn test_dst4_accuracy() {
    dct_test_with_known_data!(DST4Naive, process_dst4, slow_dst4, known_values_dst4);
}


#[test]
fn test_mdct_accuracy() {
    for curent_window_fn in &[window_fn::one, window_fn::mp3, window_fn::vorbis] {
        for len in 1..10 {
            test_mdct::planned_matches_naive(len * 2, curent_window_fn);
        }
        for &len in &[50, 52] {
            test_mdct::planned_matches_naive(len * 2, curent_window_fn);
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
            test_imdct::planned_matches_naive(len * 2, curent_window_fn);
        }
        for &len in &[50, 52] {
            test_imdct::planned_matches_naive(len * 2, curent_window_fn);
        }
    }
}
