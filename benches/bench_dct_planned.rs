#![feature(test)]
extern crate test;
extern crate num;
extern crate rust_dct;

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_planned(b: &mut Bencher, len: usize) {

    let mut planner = rust_dct::DCTPlanner::new();
    let mut dct = planner.plan_dct1(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

// small mixed composites times a large prime
#[bench] fn dct1_composite_00256(b: &mut Bencher) { bench_dct1_planned(b,  256); }
#[bench] fn dct1_composite_65536(b: &mut Bencher) { bench_dct1_planned(b,  65536); }



/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_planned(b: &mut Bencher, len: usize) {

    let mut planner = rust_dct::DCTPlanner::new();
    let mut dct = planner.plan_dct2(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

// small mixed composites times a large prime
#[bench] fn dct2_composite_00256(b: &mut Bencher) { bench_dct2_planned(b,  256); }
#[bench] fn dct2_composite_65536(b: &mut Bencher) { bench_dct2_planned(b,  65536); }




/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_planned(b: &mut Bencher, len: usize) {

    let mut planner = rust_dct::DCTPlanner::new();
    let mut dct = planner.plan_dct3(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

// small mixed composites times a large prime
#[bench] fn dct3_composite_00256(b: &mut Bencher) { bench_dct3_planned(b,  256); }
#[bench] fn dct3_composite_65536(b: &mut Bencher) { bench_dct3_planned(b,  65536); }




/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_planned(b: &mut Bencher, len: usize) {

    let mut planner = rust_dct::DCTPlanner::new();
    let mut dct = planner.plan_dct4(len);

    let mut signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct.process(&mut signal, &mut spectrum);} );
}

// small mixed composites times a large prime
#[bench] fn dct4_composite_00256(b: &mut Bencher) { bench_dct4_planned(b,  256); }
#[bench] fn dct4_composite_65536(b: &mut Bencher) { bench_dct4_planned(b,  65536); }