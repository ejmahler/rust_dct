#![feature(test)]
extern crate test;
extern crate num;
extern crate rust_dct;

use rust_dct::rustfft::Planner;
use rust_dct::dct1::{DCT1, DCT1ViaFFT};
use rust_dct::dct2::{DCT2, DCT2ViaFFT};
use rust_dct::dct3::{DCT3, DCT3ViaFFT};
use rust_dct::dct4::{DCT4, DCT4ViaFFT};

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_fft(b: &mut Bencher, len: usize) {

    let mut planner = Planner::new(false);
    let mut dct = DCT1ViaFFT::new(planner.plan_fft((len - 1) * 2));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

#[bench] fn dct1_fft_0024(b: &mut Bencher) { bench_dct1_fft(b,   24); }
#[bench] fn dct1_fft_0025(b: &mut Bencher) { bench_dct1_fft(b,   25); }
#[bench] fn dct1_fft_0026(b: &mut Bencher) { bench_dct1_fft(b,   26); }



/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_fft(b: &mut Bencher, len: usize) {

    let mut planner = Planner::new(false);
    let mut dct = DCT2ViaFFT::new(planner.plan_fft(len));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

#[bench] fn dct2_fft_003(b: &mut Bencher) { bench_dct2_fft(b,   3); }
#[bench] fn dct2_fft_004(b: &mut Bencher) { bench_dct2_fft(b,   4); }
#[bench] fn dct2_fft_005(b: &mut Bencher) { bench_dct2_fft(b,   5); }
#[bench] fn dct2_fft_006(b: &mut Bencher) { bench_dct2_fft(b,   6); }





/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_fft(b: &mut Bencher, len: usize) {

    let mut planner = Planner::new(false);
    let mut dct = DCT3ViaFFT::new(planner.plan_fft(len));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

#[bench] fn dct3_fft_003(b: &mut Bencher) { bench_dct3_fft(b,   3); }
#[bench] fn dct3_fft_004(b: &mut Bencher) { bench_dct3_fft(b,   4); }
#[bench] fn dct3_fft_005(b: &mut Bencher) { bench_dct3_fft(b,   5); }
#[bench] fn dct3_fft_006(b: &mut Bencher) { bench_dct3_fft(b,   6); }




/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_fft(b: &mut Bencher, len: usize) {

    let mut planner = Planner::new(false);
    let mut dct = DCT4ViaFFT::new(planner.plan_fft(len * 4));

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

#[bench] fn dct4_fft_0100(b: &mut Bencher) { bench_dct4_fft(b,  100); }
#[bench] fn dct4_fft_0101(b: &mut Bencher) { bench_dct4_fft(b,  101); }
#[bench] fn dct4_fft_0110(b: &mut Bencher) { bench_dct4_fft(b,  110); }
#[bench] fn dct4_fft_0111(b: &mut Bencher) { bench_dct4_fft(b,  111); }
#[bench] fn dct4_fft_0120(b: &mut Bencher) { bench_dct4_fft(b,  120); }
#[bench] fn dct4_fft_0121(b: &mut Bencher) { bench_dct4_fft(b,  121); }
#[bench] fn dct4_fft_0130(b: &mut Bencher) { bench_dct4_fft(b,  130); }
#[bench] fn dct4_fft_0131(b: &mut Bencher) { bench_dct4_fft(b,  131); }
#[bench] fn dct4_fft_0140(b: &mut Bencher) { bench_dct4_fft(b,  140); }
#[bench] fn dct4_fft_0141(b: &mut Bencher) { bench_dct4_fft(b,  141); }