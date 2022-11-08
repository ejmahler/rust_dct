use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};
use rustdct::num_traits::{Float, FromPrimitive};

pub mod known_data;
pub mod reference_impls;

#[macro_use]
pub mod macros;

pub fn fuzzy_cmp<T: Float>(a: T, b: T, tolerance: T) -> bool {
    a >= b - tolerance && a <= b + tolerance
}

pub fn compare_float_vectors<T: Float + FromPrimitive>(expected: &[T], observed: &[T]) -> bool {
    assert_eq!(expected.len(), observed.len());

    let tolerance = T::from_f64(0.001).unwrap();

    for i in 0..expected.len() {
        if !fuzzy_cmp(observed[i], expected[i], tolerance) {
            return false;
        }
    }
    true
}

pub fn random_signal<T: Float + FromPrimitive>(length: usize) -> Vec<T> {
    let mut sig = Vec::with_capacity(length);
    let distribution = Uniform::new(0.0, 10.0);

    let seed: [u8; 32] = [
        1, 5, 6, 7, 1, 5, 3, 7, 4, 2, 6, 2, 6, 1, 5, 6, 7, 1, 5, 3, 7, 4, 2, 6, 2, 6, 1, 5, 1, 0,
        1, 7,
    ];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    for _ in 0..length {
        sig.push(T::from_f64(distribution.sample(&mut rng)).unwrap());
    }
    return sig;
}
