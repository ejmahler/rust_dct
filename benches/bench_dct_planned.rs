#![feature(test)]
extern crate rustdct;
extern crate test;

use test::Bencher;

/// Times just the DCT1 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct1_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustdct::DctPlanner::new();
    let dct = planner.plan_dct1(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct1_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct1_planned_00256(b: &mut Bencher) {
    bench_dct1_planned(b, 256);
}
#[bench]
fn dct1_planned_65536(b: &mut Bencher) {
    bench_dct1_planned(b, 65536);
}

/// Times just the DCT2 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct2_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustdct::DctPlanner::new();
    let dct = planner.plan_dct2(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct2_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct2_planned_00256(b: &mut Bencher) {
    bench_dct2_planned(b, 256);
}
#[bench]
fn dct2_planned_65536(b: &mut Bencher) {
    bench_dct2_planned(b, 65536);
}

/// Times just the DCT3 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct3_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustdct::DctPlanner::new();
    let dct = planner.plan_dct3(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct3_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct3_planned_00256(b: &mut Bencher) {
    bench_dct3_planned(b, 256);
}
#[bench]
fn dct3_planned_65536(b: &mut Bencher) {
    bench_dct3_planned(b, 65536);
}

/// Times just the DCT4 execution (not allocation and pre-calculation)
/// for a given length
fn bench_dct4_planned(b: &mut Bencher, len: usize) {
    let mut planner = rustdct::DctPlanner::new();
    let dct = planner.plan_dct4(len);

    let mut buffer = vec![0_f32; len];
    let mut scratch = vec![0_f32; dct.get_scratch_len()];
    b.iter(|| {
        dct.process_dct4_with_scratch(&mut buffer, &mut scratch);
    });
}

#[bench]
fn dct4_planned_0000256(b: &mut Bencher) {
    bench_dct4_planned(b, 256);
}
#[bench]
fn dct4_planned_0999999(b: &mut Bencher) {
    bench_dct4_planned(b, 999999);
}
#[bench]
fn dct4_planned_1000000(b: &mut Bencher) {
    bench_dct4_planned(b, 1000000);
}
