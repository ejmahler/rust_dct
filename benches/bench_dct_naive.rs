#![feature(test)]
extern crate rustdct;
extern crate test;

use rustdct::mdct::{window_fn, Mdct, MdctNaive};
use rustdct::{
    algorithm::{Dct1Naive, Dst6And7Naive, Type2And3Naive, Type4Naive},
    RequiredScratch,
};
use rustdct::{Dct1, Dct2, Dct3, Dct4, Dst6, Dst7};

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_naive(b: &mut Bencher, len: usize) {
    let dct = Dct1Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct1_with_scratch(&mut buffer, &mut scratch);
    });
}
#[bench]
fn dct1_naive_002(b: &mut Bencher) {
    bench_dct1_naive(b, 2);
}
#[bench]
fn dct1_naive_004(b: &mut Bencher) {
    bench_dct1_naive(b, 4);
}
#[bench]
fn dct1_naive_006(b: &mut Bencher) {
    bench_dct1_naive(b, 6);
}
#[bench]
fn dct1_naive_008(b: &mut Bencher) {
    bench_dct1_naive(b, 8);
}
#[bench]
fn dct1_naive_010(b: &mut Bencher) {
    bench_dct1_naive(b, 10);
}

/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_naive(b: &mut Bencher, len: usize) {
    let dct = Type2And3Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct2_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct2_naive_06(b: &mut Bencher) {
    bench_dct2_naive(b, 6);
}
#[bench]
fn dct2_naive_05(b: &mut Bencher) {
    bench_dct2_naive(b, 5);
}
#[bench]
fn dct2_naive_04(b: &mut Bencher) {
    bench_dct2_naive(b, 4);
}
#[bench]
fn dct2_naive_03(b: &mut Bencher) {
    bench_dct2_naive(b, 3);
}
#[bench]
fn dct2_naive_02(b: &mut Bencher) {
    bench_dct2_naive(b, 2);
}
#[bench]
fn dct2_naive_01(b: &mut Bencher) {
    bench_dct2_naive(b, 1);
}

/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_naive(b: &mut Bencher, len: usize) {
    let dct = Type2And3Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct3_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct3_naive_0002(b: &mut Bencher) {
    bench_dct3_naive(b, 2);
}
#[bench]
fn dct3_naive_0003(b: &mut Bencher) {
    bench_dct3_naive(b, 3);
}
#[bench]
fn dct3_naive_0004(b: &mut Bencher) {
    bench_dct3_naive(b, 4);
}
#[bench]
fn dct3_naive_0005(b: &mut Bencher) {
    bench_dct3_naive(b, 5);
}
#[bench]
fn dct3_naive_0006(b: &mut Bencher) {
    bench_dct3_naive(b, 6);
}

/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_naive(b: &mut Bencher, len: usize) {
    let dct = Type4Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct4_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct4_even_naive_02(b: &mut Bencher) {
    bench_dct4_naive(b, 2);
}
#[bench]
fn dct4_even_naive_04(b: &mut Bencher) {
    bench_dct4_naive(b, 4);
}
#[bench]
fn dct4_even_naive_06(b: &mut Bencher) {
    bench_dct4_naive(b, 6);
}
#[bench]
fn dct4_even_naive_08(b: &mut Bencher) {
    bench_dct4_naive(b, 8);
}
#[bench]
fn dct4_even_naive_10(b: &mut Bencher) {
    bench_dct4_naive(b, 10);
}

#[bench]
fn dct4_odd_naive_01(b: &mut Bencher) {
    bench_dct4_naive(b, 1);
}
#[bench]
fn dct4_odd_naive_03(b: &mut Bencher) {
    bench_dct4_naive(b, 3);
}
#[bench]
fn dct4_odd_naive_05(b: &mut Bencher) {
    bench_dct4_naive(b, 5);
}
#[bench]
fn dct4_odd_naive_07(b: &mut Bencher) {
    bench_dct4_naive(b, 7);
}
#[bench]
fn dct4_odd_naive_09(b: &mut Bencher) {
    bench_dct4_naive(b, 9);
}

/// Times just the MDCT execution (not allocation and pre-calculation)
/// for a given length
fn bench_mdct_naive(b: &mut Bencher, len: usize) {
    let dct = MdctNaive::new(len, window_fn::mp3);

    let input = vec![0_f32; len * 2];
    let (input_a, input_b) = input.split_at(len);
    let mut output = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];

    b.iter(|| {
        dct.process_mdct_with_scratch(input_a, input_b, &mut output, &mut scratch);
    });
}

#[bench]
fn mdct_naive_02(b: &mut Bencher) {
    bench_mdct_naive(b, 2);
}
#[bench]
fn mdct_naive_04(b: &mut Bencher) {
    bench_mdct_naive(b, 4);
}
#[bench]
fn mdct_naive_06(b: &mut Bencher) {
    bench_mdct_naive(b, 6);
}
#[bench]
fn mdct_naive_08(b: &mut Bencher) {
    bench_mdct_naive(b, 8);
}
#[bench]
fn mdct_naive_10(b: &mut Bencher) {
    bench_mdct_naive(b, 10);
}
#[bench]
fn mdct_naive_12(b: &mut Bencher) {
    bench_mdct_naive(b, 12);
}

/// Times just the IMDCT execution (not allocation and pre-calculation)
/// for a given length
fn bench_imdct_naive(b: &mut Bencher, len: usize) {
    let dct = MdctNaive::new(len, window_fn::mp3);

    let input = vec![0_f32; len];
    let mut output = vec![0_f32; len * 2];
    let (output_a, output_b) = output.split_at_mut(len);
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_imdct_with_scratch(&input, output_a, output_b, &mut scratch);
    });
}

#[bench]
fn imdct_naive_02(b: &mut Bencher) {
    bench_imdct_naive(b, 2);
}
#[bench]
fn imdct_naive_04(b: &mut Bencher) {
    bench_imdct_naive(b, 4);
}
#[bench]
fn imdct_naive_06(b: &mut Bencher) {
    bench_imdct_naive(b, 6);
}
#[bench]
fn imdct_naive_08(b: &mut Bencher) {
    bench_imdct_naive(b, 8);
}
#[bench]
fn imdct_naive_10(b: &mut Bencher) {
    bench_imdct_naive(b, 10);
}
#[bench]
fn imdct_naive_12(b: &mut Bencher) {
    bench_imdct_naive(b, 12);
}

/// Times just the DST6 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dst6_naive(b: &mut Bencher, len: usize) {
    let dct = Dst6And7Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dst6_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dst6_even_naive_10(b: &mut Bencher) {
    bench_dst6_naive(b, 10);
}
#[bench]
fn dst6_even_naive_11(b: &mut Bencher) {
    bench_dst6_naive(b, 11);
}
#[bench]
fn dst6_even_naive_12(b: &mut Bencher) {
    bench_dst6_naive(b, 12);
}
#[bench]
fn dst6_even_naive_13(b: &mut Bencher) {
    bench_dst6_naive(b, 13);
}
#[bench]
fn dst6_even_naive_14(b: &mut Bencher) {
    bench_dst6_naive(b, 14);
}
#[bench]
fn dst6_even_naive_15(b: &mut Bencher) {
    bench_dst6_naive(b, 15);
}
#[bench]
fn dst6_even_naive_16(b: &mut Bencher) {
    bench_dst6_naive(b, 16);
}
#[bench]
fn dst6_even_naive_17(b: &mut Bencher) {
    bench_dst6_naive(b, 17);
}
#[bench]
fn dst6_even_naive_18(b: &mut Bencher) {
    bench_dst6_naive(b, 18);
}
#[bench]
fn dst6_even_naive_19(b: &mut Bencher) {
    bench_dst6_naive(b, 19);
}
#[bench]
fn dst6_even_naive_20(b: &mut Bencher) {
    bench_dst6_naive(b, 20);
}
#[bench]
fn dst6_even_naive_21(b: &mut Bencher) {
    bench_dst6_naive(b, 21);
}
#[bench]
fn dst6_even_naive_22(b: &mut Bencher) {
    bench_dst6_naive(b, 22);
}
#[bench]
fn dst6_even_naive_23(b: &mut Bencher) {
    bench_dst6_naive(b, 23);
}
#[bench]
fn dst6_even_naive_24(b: &mut Bencher) {
    bench_dst6_naive(b, 24);
}
#[bench]
fn dst6_even_naive_25(b: &mut Bencher) {
    bench_dst6_naive(b, 25);
}
#[bench]
fn dst6_even_naive_26(b: &mut Bencher) {
    bench_dst6_naive(b, 26);
}
#[bench]
fn dst6_even_naive_27(b: &mut Bencher) {
    bench_dst6_naive(b, 27);
}
#[bench]
fn dst6_even_naive_28(b: &mut Bencher) {
    bench_dst6_naive(b, 28);
}
#[bench]
fn dst6_even_naive_29(b: &mut Bencher) {
    bench_dst6_naive(b, 29);
}
#[bench]
fn dst6_even_naive_30(b: &mut Bencher) {
    bench_dst6_naive(b, 30);
}
#[bench]
fn dst6_even_naive_31(b: &mut Bencher) {
    bench_dst6_naive(b, 31);
}
#[bench]
fn dst6_even_naive_32(b: &mut Bencher) {
    bench_dst6_naive(b, 32);
}
#[bench]
fn dst6_even_naive_33(b: &mut Bencher) {
    bench_dst6_naive(b, 33);
}
#[bench]
fn dst6_even_naive_34(b: &mut Bencher) {
    bench_dst6_naive(b, 34);
}
#[bench]
fn dst6_even_naive_35(b: &mut Bencher) {
    bench_dst6_naive(b, 35);
}
#[bench]
fn dst6_even_naive_36(b: &mut Bencher) {
    bench_dst6_naive(b, 36);
}
#[bench]
fn dst6_even_naive_37(b: &mut Bencher) {
    bench_dst6_naive(b, 37);
}
#[bench]
fn dst6_even_naive_38(b: &mut Bencher) {
    bench_dst6_naive(b, 38);
}
#[bench]
fn dst6_even_naive_39(b: &mut Bencher) {
    bench_dst6_naive(b, 39);
}

/// Times just the DST7 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dst7_naive(b: &mut Bencher, len: usize) {
    let dct = Dst6And7Naive::new(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dst7_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dst7_even_naive_10(b: &mut Bencher) {
    bench_dst7_naive(b, 10);
}
#[bench]
fn dst7_even_naive_11(b: &mut Bencher) {
    bench_dst7_naive(b, 11);
}
#[bench]
fn dst7_even_naive_12(b: &mut Bencher) {
    bench_dst7_naive(b, 12);
}
#[bench]
fn dst7_even_naive_13(b: &mut Bencher) {
    bench_dst7_naive(b, 13);
}
#[bench]
fn dst7_even_naive_14(b: &mut Bencher) {
    bench_dst7_naive(b, 14);
}
#[bench]
fn dst7_even_naive_15(b: &mut Bencher) {
    bench_dst7_naive(b, 15);
}
#[bench]
fn dst7_even_naive_16(b: &mut Bencher) {
    bench_dst7_naive(b, 16);
}
#[bench]
fn dst7_even_naive_17(b: &mut Bencher) {
    bench_dst7_naive(b, 17);
}
#[bench]
fn dst7_even_naive_18(b: &mut Bencher) {
    bench_dst7_naive(b, 18);
}
#[bench]
fn dst7_even_naive_19(b: &mut Bencher) {
    bench_dst7_naive(b, 19);
}
#[bench]
fn dst7_even_naive_20(b: &mut Bencher) {
    bench_dst7_naive(b, 20);
}
#[bench]
fn dst7_even_naive_21(b: &mut Bencher) {
    bench_dst7_naive(b, 21);
}
#[bench]
fn dst7_even_naive_22(b: &mut Bencher) {
    bench_dst7_naive(b, 22);
}
#[bench]
fn dst7_even_naive_23(b: &mut Bencher) {
    bench_dst7_naive(b, 23);
}
#[bench]
fn dst7_even_naive_24(b: &mut Bencher) {
    bench_dst7_naive(b, 24);
}
#[bench]
fn dst7_even_naive_25(b: &mut Bencher) {
    bench_dst7_naive(b, 25);
}
#[bench]
fn dst7_even_naive_26(b: &mut Bencher) {
    bench_dst7_naive(b, 26);
}
#[bench]
fn dst7_even_naive_27(b: &mut Bencher) {
    bench_dst7_naive(b, 27);
}
#[bench]
fn dst7_even_naive_28(b: &mut Bencher) {
    bench_dst7_naive(b, 28);
}
#[bench]
fn dst7_even_naive_29(b: &mut Bencher) {
    bench_dst7_naive(b, 29);
}
#[bench]
fn dst7_even_naive_30(b: &mut Bencher) {
    bench_dst7_naive(b, 30);
}
#[bench]
fn dst7_even_naive_31(b: &mut Bencher) {
    bench_dst7_naive(b, 31);
}
#[bench]
fn dst7_even_naive_32(b: &mut Bencher) {
    bench_dst7_naive(b, 32);
}
#[bench]
fn dst7_even_naive_33(b: &mut Bencher) {
    bench_dst7_naive(b, 33);
}
#[bench]
fn dst7_even_naive_34(b: &mut Bencher) {
    bench_dst7_naive(b, 34);
}
#[bench]
fn dst7_even_naive_35(b: &mut Bencher) {
    bench_dst7_naive(b, 35);
}
#[bench]
fn dst7_even_naive_36(b: &mut Bencher) {
    bench_dst7_naive(b, 36);
}
#[bench]
fn dst7_even_naive_37(b: &mut Bencher) {
    bench_dst7_naive(b, 37);
}
#[bench]
fn dst7_even_naive_38(b: &mut Bencher) {
    bench_dst7_naive(b, 38);
}
#[bench]
fn dst7_even_naive_39(b: &mut Bencher) {
    bench_dst7_naive(b, 39);
}
