#![feature(test)]
extern crate test;
extern crate num;
extern crate rust_dct;

use test::Bencher;

/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4(b: &mut Bencher, len: usize) {

    let mut planner = rust_dct::DCTPlanner::new();
    let mut dct4 = planner.plan_dct4(len);

    let signal = vec![0_f32; len];
    let mut spectrum = signal.clone();
    b.iter(|| {dct4.process(&signal, &mut spectrum);} );
}

#[bench] fn dct4_p2_00064(b: &mut Bencher) { bench_dct4(b,   128); }
#[bench] fn dct4_p2_65536(b: &mut Bencher) { bench_dct4(b, 65536); }

// Powers of 7
#[bench] fn dct4_p7_00343(b: &mut Bencher) { bench_dct4(b,   343); }
#[bench] fn dct4_p7_16807(b: &mut Bencher) { bench_dct4(b, 16807); }

// Prime lengths
#[bench] fn dct4_prime_0019(b: &mut Bencher) { bench_dct4(b, 19); }
#[bench] fn dct4_prime_2017(b: &mut Bencher) { bench_dct4(b, 2017); }

// small mixed composites times a large prime
#[bench] fn dct4_composite_30270(b: &mut Bencher) { bench_dct4(b,  30270); }