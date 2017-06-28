#![feature(test)]
extern crate test;
extern crate rustdct;

use rustdct::rustfft::FFTplanner;
use rustdct::DCTplanner;
use rustdct::dct1::{DCT1, DCT1ViaFFT};
use rustdct::dct2::{DCT2, DCT2ViaFFT};
use rustdct::dct3::{DCT3, DCT3ViaFFT};
use rustdct::dct4::{DCT4, DCT4ViaDCT3, DCT4ViaFFTOdd};
use rustdct::mdct::{MDCT, IMDCT, MDCTViaDCT4, IMDCTViaDCT4, window_fn};

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_fft(b: &mut Bencher, len: usize) {

    let mut planner = FFTplanner::new(false);
    let mut dct = DCT1ViaFFT::new(planner.plan_fft((len - 1) * 2));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
}

#[bench]
fn dct1_fft_0024(b: &mut Bencher) {
    bench_dct1_fft(b, 24);
}
#[bench]
fn dct1_fft_0025(b: &mut Bencher) {
    bench_dct1_fft(b, 25);
}
#[bench]
fn dct1_fft_0026(b: &mut Bencher) {
    bench_dct1_fft(b, 26);
}



/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_fft(b: &mut Bencher, len: usize) {

    let mut planner = FFTplanner::new(false);
    let mut dct = DCT2ViaFFT::new(planner.plan_fft(len));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
}

#[bench]
fn dct2_fft_003(b: &mut Bencher) {
    bench_dct2_fft(b, 3);
}
#[bench]
fn dct2_fft_004(b: &mut Bencher) {
    bench_dct2_fft(b, 4);
}
#[bench]
fn dct2_fft_005(b: &mut Bencher) {
    bench_dct2_fft(b, 5);
}
#[bench]
fn dct2_fft_006(b: &mut Bencher) {
    bench_dct2_fft(b, 6);
}





/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_fft(b: &mut Bencher, len: usize) {

    let mut planner = FFTplanner::new(false);
    let mut dct = DCT3ViaFFT::new(planner.plan_fft(len));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
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



/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_via_dct3(b: &mut Bencher, len: usize) {

    let mut planner = DCTplanner::new();
    let inner_dct3 = planner.plan_dct3(len / 2);
    let mut dct = DCT4ViaDCT3::new(inner_dct3);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
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

    let mut planner = FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len);
    let mut dct = DCT4ViaFFTOdd::new(inner_fft);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| { dct.process(&mut signal, &mut spectrum); });
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

    let mut planner = DCTplanner::new();
    let mut dct = MDCTViaDCT4::new(planner.plan_dct4(len), window_fn::mp3);

    let signal = vec![0_f32; len * 2];
    let mut spectrum = vec![0_f32; len];
    b.iter(|| { dct.process(&signal, &mut spectrum); });
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

    let mut planner = DCTplanner::new();
    let mut dct = IMDCTViaDCT4::new(planner.plan_dct4(len), window_fn::mp3);

    let signal = vec![0_f32; len];
    let mut spectrum = vec![0_f32; len * 2];
    b.iter(|| { dct.process(&signal, &mut spectrum); });
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
