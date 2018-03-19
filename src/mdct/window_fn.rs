use std::f64;

use common;

/// MP3 window function for MDCT
pub fn mp3<T: common::DCTnum>(len: usize) -> Vec<T> {
    let constant_term = f64::consts::PI / len as f64;

    (0..len)
        .map(|n| (constant_term * (n as f64 + 0.5f64)).sin())
        .map(|w| T::from_f64(w).unwrap())
        .collect()
}

/// Ogg Vorbis window function for MDCT
pub fn vorbis<T: common::DCTnum>(len: usize) -> Vec<T> {
    let constant_term = f64::consts::PI / len as f64;

    (0..len)
        .map(|n| {
            let inner_sin = (constant_term * (n as f64 + 0.5f64)).sin();

            (f64::consts::PI * 0.5f64 * inner_sin * inner_sin).sin()
        })
        .map(|w| T::from_f64(w).unwrap())
        .collect()
}

/// MDCT window function which is all ones (IE, no windowing will be applied)
pub fn one<T: common::DCTnum>(len: usize) -> Vec<T> {
    (0..len).map(|_| T::one()).collect()
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::fuzzy_cmp;

    /// Verify that each of the built-in window functions does what we expect
    #[test]
    fn test_window_fns() {
        for test_fn in &[mp3, vorbis] {
            for half_size in 1..20 {
                let evaluated_window: Vec<f32> = test_fn(half_size * 2);

                //verify that for all i from 0 to half_size, window[i]^2 + window[i+half_size]^2 == 1
                //also known as the "Princen-Bradley condition"
                for i in 0..half_size {
                    let first = evaluated_window[i];
                    let second = evaluated_window[i + half_size];
                    assert!(fuzzy_cmp(first * first + second * second, 1f32, 0.001f32));
                }
            }
        }
    }
}
