#![feature(test)]
extern crate test;
extern crate rustdct;

use rustdct::dct1::{DCT1, DCT1Naive};
use rustdct::dct2::{DCT2, DCT2Naive};
use rustdct::dct3::{DCT3, DCT3Naive};
use rustdct::dct4::{DCT4, DCT4Naive};
use rustdct::mdct::{MDCT, IMDCT, MDCTNaive, IMDCTNaive, window_fn};

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_naive(b: &mut Bencher, len: usize) {

    let mut dct = DCT1Naive::new(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
}

#[bench]
fn dct1_naive_0024(b: &mut Bencher) {
    bench_dct1_naive(b, 24);
}
#[bench]
fn dct1_naive_0025(b: &mut Bencher) {
    bench_dct1_naive(b, 25);
}
#[bench]
fn dct1_naive_0026(b: &mut Bencher) {
    bench_dct1_naive(b, 26);
}



/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_naive(b: &mut Bencher, len: usize) {

    let mut dct = DCT2Naive::new(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
}

#[bench]
fn dct2_naive_0003(b: &mut Bencher) {
    bench_dct2_naive(b, 3);
}
#[bench]
fn dct2_naive_0004(b: &mut Bencher) {
    bench_dct2_naive(b, 4);
}
#[bench]
fn dct2_naive_0005(b: &mut Bencher) {
    bench_dct2_naive(b, 5);
}
#[bench]
fn dct2_naive_0006(b: &mut Bencher) {
    bench_dct2_naive(b, 6);
}





/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_naive(b: &mut Bencher, len: usize) {

    let mut dct = DCT3Naive::new(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
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

    let mut dct = DCT4Naive::new(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
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
fn dct4_odd_naive_0135(b: &mut Bencher) {
    bench_dct4_naive(b, 135);
}
#[bench]
fn dct4_odd_naive_0141(b: &mut Bencher) {
    bench_dct4_naive(b, 141);
}
#[bench]
fn dct4_odd_naive_0145(b: &mut Bencher) {
    bench_dct4_naive(b, 145);
}
#[bench]
fn dct4_odd_naive_0151(b: &mut Bencher) {
    bench_dct4_naive(b, 151);
}
#[bench]
fn dct4_odd_naive_0155(b: &mut Bencher) {
    bench_dct4_naive(b, 155);
}


/// Times just the MDCT execution (not allocation and pre-calculation)
/// for a given length
fn bench_mdct_naive(b: &mut Bencher, len: usize) {

    let mut dct = MDCTNaive::new(len, window_fn::mp3);

    let signal = vec![0_f32; len * 2];
    let mut spectrum = vec![0_f32; len];
    b.iter(|| { dct.process(&signal, &mut spectrum); });
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

    let mut dct = IMDCTNaive::new(len, window_fn::mp3);

    let signal = vec![0_f32; len];
    let mut spectrum = vec![0_f32; len * 2];
    b.iter(|| { dct.process(&signal, &mut spectrum); });
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
