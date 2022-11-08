//! This example is meant to be used for inspecting the generated assembly.
//! This can be interesting when working with simd intrinsics, or trying to reduce code size
//!
//! To use:
//! - Mark the function that should be investigated with `#[inline(never)]`.
//! - If needed, add any required feature to the function, for example `#[target_feature(enable = "sse4.1")]`
//! - Change the code below to use the changed function.
//!   Currently it is set up to call a generic planned DCT, which will cause all dct code to be compiled
//! - Ask rustc to output assembly code:
//!   `cargo rustc --release --example asmtest -- --emit=asm`
//! - This will create a file at `target/release/examples/asmtest-0123456789abcdef.s` (with a random number in the filename).
//! - Open this file and search for the function.

use rustdct::DctPlanner;

fn main() {
    let mut planner = DctPlanner::new();
    let dct = planner.plan_dct2(4);

    let mut buffer: Vec<f32> = vec![0.0; dct.len()];
    dct.process_dct2(&mut buffer);
}
