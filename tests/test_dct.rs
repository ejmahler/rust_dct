extern crate rand;
extern crate rustdct;

#[macro_use]
mod common;

use rustdct::algorithm::{
    Dct1Naive, Dct5Naive, Dct6And7Naive, Dct8Naive, Dst1Naive, Dst5Naive, Dst6And7Naive, Dst8Naive,
    Type2And3Naive, Type4Naive,
};
use rustdct::mdct::window_fn;
use rustdct::DctPlanner;
use rustdct::{
    Dct1, Dct2, Dct3, Dct4, Dct5, Dct6, Dct7, Dct8, Dst1, Dst2, Dst3, Dst4, Dst5, Dst6, Dst7, Dst8,
};

use crate::common::known_data::*;
use crate::common::macros::test_mdct;
use crate::common::reference_impls::*;
use crate::common::{compare_float_vectors, random_signal};

use std::f32;

fn inverse_scale_dct1(len: usize) -> f64 {
    2.0 / (len - 1) as f64
}
fn inverse_scale_dst1(len: usize) -> f64 {
    2.0 / (len + 1) as f64
}
fn inverse_scale_normal(len: usize) -> f64 {
    2.0 / len as f64
}
fn inverse_scale_minushalf(len: usize) -> f64 {
    2.0 / (len as f64 - 0.5)
}
fn inverse_scale_plushalf(len: usize) -> f64 {
    2.0 / (len as f64 + 0.5)
}

#[test]
fn test_dct1_accuracy() {
    dct_test_with_known_data!(reference_dct1, Dct1Naive, process_dct1, known_values_dct1);
    dct_test_with_planner!(reference_dct1, Dct1Naive, process_dct1, plan_dct1, 2);
    dct_test_inverse!(reference_dct1, reference_dct1, inverse_scale_dct1, 2);
}
#[test]
fn test_dct2_accuracy() {
    dct_test_with_known_data!(
        reference_dct2,
        Type2And3Naive,
        process_dct2,
        known_values_dct2
    );
    dct_test_with_planner!(reference_dct2, Type2And3Naive, process_dct2, plan_dct2, 1);
    dct_test_inverse!(reference_dct2, reference_dct3, inverse_scale_normal, 1);
}
#[test]
fn test_dct3_accuracy() {
    dct_test_with_known_data!(
        reference_dct3,
        Type2And3Naive,
        process_dct3,
        known_values_dct3
    );
    dct_test_with_planner!(reference_dct3, Type2And3Naive, process_dct3, plan_dct3, 1);
    dct_test_inverse!(reference_dct3, reference_dct2, inverse_scale_normal, 1);
}
#[test]
fn test_dct4_accuracy() {
    dct_test_with_known_data!(reference_dct4, Type4Naive, process_dct4, known_values_dct4);
    dct_test_with_planner!(reference_dct4, Type4Naive, process_dct4, plan_dct4, 1);
    dct_test_inverse!(reference_dct4, reference_dct4, inverse_scale_normal, 1);
}
#[test]
fn test_dct5_accuracy() {
    dct_test_with_planner!(reference_dct5, Dct5Naive, process_dct5, plan_dct5, 1);
    dct_test_inverse!(reference_dct5, reference_dct5, inverse_scale_minushalf, 1);
}
#[test]
fn test_dct6_accuracy() {
    dct_test_with_planner!(reference_dct6, Dct6And7Naive, process_dct6, plan_dct6, 1);
    dct_test_inverse!(reference_dct6, reference_dct7, inverse_scale_minushalf, 1);
}
#[test]
fn test_dct7_accuracy() {
    dct_test_with_planner!(reference_dct7, Dct6And7Naive, process_dct7, plan_dct7, 1);
    dct_test_inverse!(reference_dct7, reference_dct6, inverse_scale_minushalf, 1);
}
#[test]
fn test_dct8_accuracy() {
    dct_test_with_planner!(reference_dct8, Dct8Naive, process_dct8, plan_dct8, 6);
    dct_test_inverse!(reference_dct8, reference_dct8, inverse_scale_plushalf, 1);
}

#[test]
fn test_dst1_accuracy() {
    dct_test_with_known_data!(reference_dst1, Dst1Naive, process_dst1, known_values_dst1);
    dct_test_with_planner!(reference_dst1, Dst1Naive, process_dst1, plan_dst1, 1);
    dct_test_inverse!(reference_dst1, reference_dst1, inverse_scale_dst1, 1);
}
#[test]
fn test_dst2_accuracy() {
    dct_test_with_known_data!(
        reference_dst2,
        Type2And3Naive,
        process_dst2,
        known_values_dst2
    );
    dct_test_with_planner!(reference_dst2, Type2And3Naive, process_dst2, plan_dst2, 1);
    dct_test_inverse!(reference_dst2, reference_dst3, inverse_scale_normal, 1);
}
#[test]
fn test_dst3_accuracy() {
    dct_test_with_known_data!(
        reference_dst3,
        Type2And3Naive,
        process_dst3,
        known_values_dst3
    );
    dct_test_with_planner!(reference_dst3, Type2And3Naive, process_dst3, plan_dst3, 1);
    dct_test_inverse!(reference_dst3, reference_dst2, inverse_scale_normal, 1);
}
#[test]
fn test_dst4_accuracy() {
    dct_test_with_known_data!(reference_dst4, Type4Naive, process_dst4, known_values_dst4);
    dct_test_with_planner!(reference_dst4, Type4Naive, process_dst4, plan_dst4, 1);
    dct_test_inverse!(reference_dst4, reference_dst4, inverse_scale_normal, 1);
}
#[test]
fn test_dst5_accuracy() {
    dct_test_with_planner!(reference_dst5, Dst5Naive, process_dst5, plan_dst5, 1);
    dct_test_inverse!(reference_dst5, reference_dst5, inverse_scale_plushalf, 1);
}
#[test]
fn test_dst6_accuracy() {
    dct_test_with_planner!(reference_dst6, Dst6And7Naive, process_dst6, plan_dst6, 1);
    dct_test_inverse!(reference_dst6, reference_dst7, inverse_scale_plushalf, 1);
}
#[test]
fn test_dst7_accuracy() {
    dct_test_with_planner!(reference_dst7, Dst6And7Naive, process_dst7, plan_dst7, 6);
    dct_test_inverse!(reference_dst7, reference_dst6, inverse_scale_plushalf, 1);
}
#[test]
fn test_dst8_accuracy() {
    dct_test_with_planner!(reference_dst8, Dst8Naive, process_dst8, plan_dst8, 6);
    dct_test_inverse!(reference_dst8, reference_dst8, inverse_scale_minushalf, 1);
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
    struct TdacTestStruct<'a> {
        name: &'static str,
        window: &'a dyn Fn(usize) -> Vec<f32>,
        scale_fn: &'a dyn Fn(usize) -> f32,
    }
    impl<'a> TdacTestStruct<'a> {
        fn new(
            name: &'static str,
            window: &'a dyn Fn(usize) -> Vec<f32>,
            scale_fn: &'a dyn Fn(usize) -> f32,
        ) -> Self {
            Self {
                name,
                window,
                scale_fn,
            }
        }
    }

    let non_window_scale = |len: usize| 1.0 / (len as f32);
    let window_scale = |len: usize| 2.0 / (len as f32);
    let invertible_scale = |_| 1.0;

    let tests = [
        TdacTestStruct::new("one", &window_fn::one, &non_window_scale),
        TdacTestStruct::new("mp3", &window_fn::mp3, &window_scale),
        TdacTestStruct::new("vorbis", &window_fn::vorbis, &window_scale),
        TdacTestStruct::new("invertible", &window_fn::invertible, &invertible_scale),
        TdacTestStruct::new(
            "mp3_invertible",
            &window_fn::mp3_invertible,
            &invertible_scale,
        ),
        TdacTestStruct::new(
            "vorbis_invertible",
            &window_fn::vorbis_invertible,
            &invertible_scale,
        ),
    ];

    for test_data in &tests {
        for i in 1..10 {
            let len = i * 2;
            println!("name: {}, len: {}", test_data.name, len);
            test_mdct::test_tdac(len, (test_data.scale_fn)(len), test_data.window);
        }
        for &i in &[50, 52] {
            let len = i * 2;
            println!("name: {}, len: {}", test_data.name, len);
            test_mdct::test_tdac(len, (test_data.scale_fn)(len), test_data.window);
        }
    }
}
