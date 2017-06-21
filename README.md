# RustDCT
[![](https://img.shields.io/crates/v/rustdct.svg)](https://crates.io/crates/rustdct)
[![](https://img.shields.io/crates/l/rustdct.svg)](https://crates.io/crates/rustdct)
[![](https://docs.rs/rustdct/badge.svg)](https://docs.rs/rustdct/)

RustDCT is a pure-Rust signal processing library that computes the most common Discrete Cosine Transforms: 

* DCT Type 1
* DCT Type 2 (Often called "the" DCT - by far the most common algorithm, used by JPEG image compression and others)
* DCT Type 3 (the inverse of the DCT type 2, also used in JPEG)
* DCT Type 4
* MDCT (Used in audio and video compression such as Ogg and MP3)

The recommended way to use RustDCT is to create a `DCTplanner` instance, then call its `plan_dct1` or `plan_dct2` or etc method. Each DCT type has its own method which will choose the best algorithm for the given size.

```rust
// Compute a DCT Type 2 of size 1234
use rustdct::DCTplanner;

let mut input:  Vec<f32> = vec![0f32; 1234];
let mut output: Vec<f32> = vec![0f32; 1234];

let mut planner = DCTplanner::new();
let mut dct = planner.plan_dct2(1234);

dct.process(&mut input, &mut output);

```
