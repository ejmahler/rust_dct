#![feature(test)]
extern crate rustdct;
extern crate test;

use std::sync::Arc;

use rustdct::algorithm::*;
use rustdct::mdct::{window_fn, Mdct, MdctViaDct4};
use rustdct::rustfft::FftPlanner;
use rustdct::DctPlanner;
use rustdct::{algorithm::type2and3_butterflies::*, RequiredScratch};
use rustdct::{Dct1, Dct2, Dct3, Dct4, Dst6, Dst7, TransformType2And3};

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_fft(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let dct = Dct1ConvertToFft::new(planner.plan_fft_forward((len - 1) * 2));

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct1_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct1_fft_002(b: &mut Bencher) {
    bench_dct1_fft(b, 2);
}
#[bench]
fn dct1_fft_004(b: &mut Bencher) {
    bench_dct1_fft(b, 4);
}
#[bench]
fn dct1_fft_006(b: &mut Bencher) {
    bench_dct1_fft(b, 6);
}
#[bench]
fn dct1_fft_008(b: &mut Bencher) {
    bench_dct1_fft(b, 8);
}
#[bench]
fn dct1_fft_010(b: &mut Bencher) {
    bench_dct1_fft(b, 10);
}

/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_fft(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let dct = Type2And3ConvertToFft::new(planner.plan_fft_forward(len));

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct2_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct2_fft_06(b: &mut Bencher) {
    bench_dct2_fft(b, 6);
}
#[bench]
fn dct2_fft_05(b: &mut Bencher) {
    bench_dct2_fft(b, 5);
}
#[bench]
fn dct2_fft_04(b: &mut Bencher) {
    bench_dct2_fft(b, 4);
}
#[bench]
fn dct2_fft_03(b: &mut Bencher) {
    bench_dct2_fft(b, 3);
}
#[bench]
fn dct2_fft_02(b: &mut Bencher) {
    bench_dct2_fft(b, 2);
}
#[bench]
fn dct2_fft_01(b: &mut Bencher) {
    bench_dct2_fft(b, 1);
}

/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_split(b: &mut Bencher, len: usize) {
    let power = len.trailing_zeros() as usize;
    let mut instances = vec![
        Arc::new(Type2And3Naive::new(1)) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly2::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly4::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly8::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly16::new()) as Arc<dyn TransformType2And3<f32>>,
    ];
    for i in instances.len()..(power + 1) {
        let dct = Arc::new(Type2And3SplitRadix::new(
            instances[i - 1].clone(),
            instances[i - 2].clone(),
        ));
        instances.push(dct);
    }

    let dct = instances[power].clone();
    assert_eq!(dct.len(), len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct2_with_scratch(&mut buffer, &mut scratch);
    });
}
#[bench]
fn dct2_power2_split_0002(b: &mut Bencher) {
    bench_dct2_split(b, 2);
}
#[bench]
fn dct2_power2_split_0004(b: &mut Bencher) {
    bench_dct2_split(b, 4);
}
#[bench]
fn dct2_power2_split_0008(b: &mut Bencher) {
    bench_dct2_split(b, 4);
}
#[bench]
fn dct2_power2_split_0016(b: &mut Bencher) {
    bench_dct2_split(b, 16);
}
#[bench]
fn dct2_power2_split_0032(b: &mut Bencher) {
    bench_dct2_split(b, 32);
}
#[bench]
fn dct2_power2_split_0064(b: &mut Bencher) {
    bench_dct2_split(b, 64);
}
#[bench]
fn dct2_power2_split_0128(b: &mut Bencher) {
    bench_dct2_split(b, 128);
}
#[bench]
fn dct2_power2_split_0256(b: &mut Bencher) {
    bench_dct2_split(b, 256);
}
#[bench]
fn dct2_power2_split_065536(b: &mut Bencher) {
    bench_dct2_split(b, 65536);
}

#[bench]
fn dct2_power2_fft_0002(b: &mut Bencher) {
    bench_dct2_fft(b, 2);
}
#[bench]
fn dct2_power2_fft_0004(b: &mut Bencher) {
    bench_dct2_fft(b, 4);
}
#[bench]
fn dct2_power2_fft_0008(b: &mut Bencher) {
    bench_dct2_fft(b, 4);
}
#[bench]
fn dct2_power2_fft_0016(b: &mut Bencher) {
    bench_dct2_fft(b, 16);
}
#[bench]
fn dct2_power2_fft_0032(b: &mut Bencher) {
    bench_dct2_fft(b, 32);
}
#[bench]
fn dct2_power2_fft_0064(b: &mut Bencher) {
    bench_dct2_fft(b, 64);
}
#[bench]
fn dct2_power2_fft_0128(b: &mut Bencher) {
    bench_dct2_fft(b, 128);
}
#[bench]
fn dct2_power2_fft_0256(b: &mut Bencher) {
    bench_dct2_fft(b, 256);
}
#[bench]
fn dct2_power2_fft_065536(b: &mut Bencher) {
    bench_dct2_fft(b, 65536);
}

/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_fft(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let dct = Type2And3ConvertToFft::new(planner.plan_fft_forward(len));

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct3_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct3_fft_003(b: &mut Bencher) {
    bench_dct3_fft(b, 3);
}
#[bench]
fn dct3_fft_004(b: &mut Bencher) {
    bench_dct3_fft(b, 4);
}
#[bench]
fn dct3_fft_005(b: &mut Bencher) {
    bench_dct3_fft(b, 5);
}
#[bench]
fn dct3_fft_006(b: &mut Bencher) {
    bench_dct3_fft(b, 6);
}

#[bench]
fn dct3_power2_fft_00004(b: &mut Bencher) {
    bench_dct3_fft(b, 4);
}
#[bench]
fn dct3_power2_fft_00008(b: &mut Bencher) {
    bench_dct3_fft(b, 8);
}
#[bench]
fn dct3_power2_fft_00016(b: &mut Bencher) {
    bench_dct3_fft(b, 16);
}
#[bench]
fn dct3_power2_fft_00032(b: &mut Bencher) {
    bench_dct3_fft(b, 32);
}
#[bench]
fn dct3_power2_fft_00064(b: &mut Bencher) {
    bench_dct3_fft(b, 64);
}
#[bench]
fn dct3_power2_fft_00256(b: &mut Bencher) {
    bench_dct3_fft(b, 256);
}
#[bench]
fn dct3_power2_fft_065536(b: &mut Bencher) {
    bench_dct3_fft(b, 65536);
}
#[bench]
fn dct3_power2_fft_16777216(b: &mut Bencher) {
    bench_dct3_fft(b, 16777216);
}

/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_split(b: &mut Bencher, len: usize) {
    let power = len.trailing_zeros() as usize;
    let mut instances = vec![
        Arc::new(Type2And3Naive::new(1)) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly2::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly4::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly8::new()) as Arc<dyn TransformType2And3<f32>>,
        Arc::new(Type2And3Butterfly16::new()) as Arc<dyn TransformType2And3<f32>>,
    ];
    for i in instances.len()..(power + 1) {
        let dct = Arc::new(Type2And3SplitRadix::new(
            instances[i - 1].clone(),
            instances[i - 2].clone(),
        ));
        instances.push(dct);
    }

    let dct = instances[power].clone();
    assert_eq!(dct.len(), len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct3_with_scratch(&mut buffer, &mut scratch);
    });
}
#[bench]
fn dct3_power2_split_0002(b: &mut Bencher) {
    bench_dct3_split(b, 4);
}
#[bench]
fn dct3_power2_split_0004(b: &mut Bencher) {
    bench_dct3_split(b, 4);
}
#[bench]
fn dct3_power2_split_0008(b: &mut Bencher) {
    bench_dct3_split(b, 4);
}
#[bench]
fn dct3_power2_split_0016(b: &mut Bencher) {
    bench_dct3_split(b, 16);
}
#[bench]
fn dct3_power2_split_0064(b: &mut Bencher) {
    bench_dct3_split(b, 64);
}
#[bench]
fn dct3_power2_split_0256(b: &mut Bencher) {
    bench_dct3_split(b, 256);
}
#[bench]
fn dct3_power2_split_065536(b: &mut Bencher) {
    bench_dct3_split(b, 65536);
}

/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_via_dct3(b: &mut Bencher, len: usize) {
    let mut planner = DctPlanner::new();
    let inner_dct3 = planner.plan_dct3(len / 2);
    let dct = Type4ConvertToType3Even::new(inner_dct3);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct4_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct4_even_via_dct3_02(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 2);
}
#[bench]
fn dct4_even_via_dct3_04(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 4);
}
#[bench]
fn dct4_even_via_dct3_06(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 6);
}
#[bench]
fn dct4_even_via_dct3_08(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 8);
}
#[bench]
fn dct4_even_via_dct3_10(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 10);
}

#[bench]
fn dct4_even_via_dct3_1000000(b: &mut Bencher) {
    bench_dct4_via_dct3(b, 1000000);
}

/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_via_fft_odd(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let inner_fft = planner.plan_fft_forward(len);
    let dct = Type4ConvertToFftOdd::new(inner_fft);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct4_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct4_odd_via_fft_01(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 1);
}
#[bench]
fn dct4_odd_via_fft_03(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 3);
}
#[bench]
fn dct4_odd_via_fft_05(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 5);
}
#[bench]
fn dct4_odd_via_fft_07(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 7);
}
#[bench]
fn dct4_odd_via_fft_09(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 9);
}
#[bench]
fn dct4_odd_via_fft_999999(b: &mut Bencher) {
    bench_dct4_via_fft_odd(b, 999999);
}

/// Times just the MDCT execution (not allocation and pre-calculation)
/// for a given length
fn bench_mdct_fft(b: &mut Bencher, len: usize) {
    let mut planner = DctPlanner::new();
    let dct = MdctViaDct4::new(planner.plan_dct4(len), window_fn::mp3);

    let input = vec![0_f32; len * 2];
    let (input_a, input_b) = input.split_at(len);
    let mut output = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];

    b.iter(|| {
        dct.process_mdct_with_scratch(input_a, input_b, &mut output, &mut scratch);
    });
}
#[bench]
fn mdct_fft_02(b: &mut Bencher) {
    bench_mdct_fft(b, 2);
}
#[bench]
fn mdct_fft_04(b: &mut Bencher) {
    bench_mdct_fft(b, 4);
}
#[bench]
fn mdct_fft_06(b: &mut Bencher) {
    bench_mdct_fft(b, 6);
}
#[bench]
fn mdct_fft_08(b: &mut Bencher) {
    bench_mdct_fft(b, 8);
}
#[bench]
fn mdct_fft_10(b: &mut Bencher) {
    bench_mdct_fft(b, 10);
}
#[bench]
fn mdct_fft_12(b: &mut Bencher) {
    bench_mdct_fft(b, 12);
}

/// Times just the IMDCT execution (not allocation and pre-calculation)
/// for a given length
fn bench_imdct_fft(b: &mut Bencher, len: usize) {
    let mut planner = DctPlanner::new();
    let dct = MdctViaDct4::new(planner.plan_dct4(len), window_fn::mp3);

    let input = vec![0_f32; len];
    let mut output = vec![0_f32; len * 2];
    let (output_a, output_b) = output.split_at_mut(len);
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_imdct_with_scratch(&input, output_a, output_b, &mut scratch);
    });
}
#[bench]
fn imdct_fft_02(b: &mut Bencher) {
    bench_imdct_fft(b, 2);
}
#[bench]
fn imdct_fft_04(b: &mut Bencher) {
    bench_imdct_fft(b, 4);
}
#[bench]
fn imdct_fft_06(b: &mut Bencher) {
    bench_imdct_fft(b, 6);
}
#[bench]
fn imdct_fft_08(b: &mut Bencher) {
    bench_imdct_fft(b, 8);
}
#[bench]
fn imdct_fft_10(b: &mut Bencher) {
    bench_imdct_fft(b, 10);
}
#[bench]
fn imdct_fft_12(b: &mut Bencher) {
    bench_imdct_fft(b, 12);
}

/// Times just the DST6 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dst6_fft(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let dct = Dst6And7ConvertToFft::new(planner.plan_fft_forward(len * 2 + 1));

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dst6_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dst6_fft_10(b: &mut Bencher) {
    bench_dst6_fft(b, 10);
}
#[bench]
fn dst6_fft_11(b: &mut Bencher) {
    bench_dst6_fft(b, 11);
}
#[bench]
fn dst6_fft_12(b: &mut Bencher) {
    bench_dst6_fft(b, 12);
}
#[bench]
fn dst6_fft_13(b: &mut Bencher) {
    bench_dst6_fft(b, 13);
}
#[bench]
fn dst6_fft_14(b: &mut Bencher) {
    bench_dst6_fft(b, 14);
}
#[bench]
fn dst6_fft_15(b: &mut Bencher) {
    bench_dst6_fft(b, 15);
}
#[bench]
fn dst6_fft_16(b: &mut Bencher) {
    bench_dst6_fft(b, 16);
}
#[bench]
fn dst6_fft_17(b: &mut Bencher) {
    bench_dst6_fft(b, 17);
}
#[bench]
fn dst6_fft_18(b: &mut Bencher) {
    bench_dst6_fft(b, 18);
}
#[bench]
fn dst6_fft_19(b: &mut Bencher) {
    bench_dst6_fft(b, 19);
}
#[bench]
fn dst6_fft_20(b: &mut Bencher) {
    bench_dst6_fft(b, 20);
}
#[bench]
fn dst6_fft_21(b: &mut Bencher) {
    bench_dst6_fft(b, 21);
}
#[bench]
fn dst6_fft_22(b: &mut Bencher) {
    bench_dst6_fft(b, 22);
}
#[bench]
fn dst6_fft_23(b: &mut Bencher) {
    bench_dst6_fft(b, 23);
}
#[bench]
fn dst6_fft_24(b: &mut Bencher) {
    bench_dst6_fft(b, 24);
}
#[bench]
fn dst6_fft_25(b: &mut Bencher) {
    bench_dst6_fft(b, 25);
}
#[bench]
fn dst6_fft_26(b: &mut Bencher) {
    bench_dst6_fft(b, 26);
}
#[bench]
fn dst6_fft_27(b: &mut Bencher) {
    bench_dst6_fft(b, 27);
}
#[bench]
fn dst6_fft_28(b: &mut Bencher) {
    bench_dst6_fft(b, 28);
}
#[bench]
fn dst6_fft_29(b: &mut Bencher) {
    bench_dst6_fft(b, 29);
}
#[bench]
fn dst6_fft_30(b: &mut Bencher) {
    bench_dst6_fft(b, 30);
}
#[bench]
fn dst6_fft_31(b: &mut Bencher) {
    bench_dst6_fft(b, 31);
}
#[bench]
fn dst6_fft_32(b: &mut Bencher) {
    bench_dst6_fft(b, 32);
}
#[bench]
fn dst6_fft_33(b: &mut Bencher) {
    bench_dst6_fft(b, 33);
}
#[bench]
fn dst6_fft_34(b: &mut Bencher) {
    bench_dst6_fft(b, 34);
}
#[bench]
fn dst6_fft_35(b: &mut Bencher) {
    bench_dst6_fft(b, 35);
}
#[bench]
fn dst6_fft_36(b: &mut Bencher) {
    bench_dst6_fft(b, 36);
}
#[bench]
fn dst6_fft_37(b: &mut Bencher) {
    bench_dst6_fft(b, 37);
}
#[bench]
fn dst6_fft_38(b: &mut Bencher) {
    bench_dst6_fft(b, 38);
}
#[bench]
fn dst6_fft_39(b: &mut Bencher) {
    bench_dst6_fft(b, 39);
}

/// Times just the DST6 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dst7_fft(b: &mut Bencher, len: usize) {
    let mut planner = FftPlanner::new();
    let dct = Dst6And7ConvertToFft::new(planner.plan_fft_forward(len * 2 + 1));

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dst7_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dst7_fft_10(b: &mut Bencher) {
    bench_dst7_fft(b, 10);
}
#[bench]
fn dst7_fft_11(b: &mut Bencher) {
    bench_dst7_fft(b, 11);
}
#[bench]
fn dst7_fft_12(b: &mut Bencher) {
    bench_dst7_fft(b, 12);
}
#[bench]
fn dst7_fft_13(b: &mut Bencher) {
    bench_dst7_fft(b, 13);
}
#[bench]
fn dst7_fft_14(b: &mut Bencher) {
    bench_dst7_fft(b, 14);
}
#[bench]
fn dst7_fft_15(b: &mut Bencher) {
    bench_dst7_fft(b, 15);
}
#[bench]
fn dst7_fft_16(b: &mut Bencher) {
    bench_dst7_fft(b, 16);
}
#[bench]
fn dst7_fft_17(b: &mut Bencher) {
    bench_dst7_fft(b, 17);
}
#[bench]
fn dst7_fft_18(b: &mut Bencher) {
    bench_dst7_fft(b, 18);
}
#[bench]
fn dst7_fft_19(b: &mut Bencher) {
    bench_dst7_fft(b, 19);
}
#[bench]
fn dst7_fft_20(b: &mut Bencher) {
    bench_dst7_fft(b, 20);
}
#[bench]
fn dst7_fft_21(b: &mut Bencher) {
    bench_dst7_fft(b, 21);
}
#[bench]
fn dst7_fft_22(b: &mut Bencher) {
    bench_dst7_fft(b, 22);
}
#[bench]
fn dst7_fft_23(b: &mut Bencher) {
    bench_dst7_fft(b, 23);
}
#[bench]
fn dst7_fft_24(b: &mut Bencher) {
    bench_dst7_fft(b, 24);
}
#[bench]
fn dst7_fft_25(b: &mut Bencher) {
    bench_dst7_fft(b, 25);
}
#[bench]
fn dst7_fft_26(b: &mut Bencher) {
    bench_dst7_fft(b, 26);
}
#[bench]
fn dst7_fft_27(b: &mut Bencher) {
    bench_dst7_fft(b, 27);
}
#[bench]
fn dst7_fft_28(b: &mut Bencher) {
    bench_dst7_fft(b, 28);
}
#[bench]
fn dst7_fft_29(b: &mut Bencher) {
    bench_dst7_fft(b, 29);
}
#[bench]
fn dst7_fft_30(b: &mut Bencher) {
    bench_dst7_fft(b, 30);
}
#[bench]
fn dst7_fft_31(b: &mut Bencher) {
    bench_dst7_fft(b, 31);
}
#[bench]
fn dst7_fft_32(b: &mut Bencher) {
    bench_dst7_fft(b, 32);
}
#[bench]
fn dst7_fft_33(b: &mut Bencher) {
    bench_dst7_fft(b, 33);
}
#[bench]
fn dst7_fft_34(b: &mut Bencher) {
    bench_dst7_fft(b, 34);
}
#[bench]
fn dst7_fft_35(b: &mut Bencher) {
    bench_dst7_fft(b, 35);
}
#[bench]
fn dst7_fft_36(b: &mut Bencher) {
    bench_dst7_fft(b, 36);
}
#[bench]
fn dst7_fft_37(b: &mut Bencher) {
    bench_dst7_fft(b, 37);
}
#[bench]
fn dst7_fft_38(b: &mut Bencher) {
    bench_dst7_fft(b, 38);
}
#[bench]
fn dst7_fft_39(b: &mut Bencher) {
    bench_dst7_fft(b, 39);
}
