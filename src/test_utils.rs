extern crate rand;

use std::f32;

use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

pub fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
    a >= b - tolerance && a <= b + tolerance
}

pub fn compare_float_vectors(expected: &[f32], observed: &[f32]) -> bool {
    assert_eq!(expected.len(), observed.len());

    let tolerance: f32 = 0.001;

    for i in 0..expected.len() {
        if !fuzzy_cmp(observed[i], expected[i], tolerance) {
            return false;
        }
    }
    true
}

pub fn random_signal(length: usize) -> Vec<f32> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Uniform::new(0.0, 10.0);

    let seed: [u8; 32] = [
        1, 5, 6, 4, 1, 5, 3, 7, 4, 2, 6, 2, 6, 4, 5, 6, 7, 1, 5, 3, 7, 4, 2, 6, 2, 6, 1, 5, 3, 0,
        1, 0,
    ];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    for _ in 0..length {
        sig.push(normal_dist.sample(&mut rng) as f32);
    }
    return sig;
}
