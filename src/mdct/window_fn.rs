
use DCTnum;
use num::Float;
use num::traits::FloatConst;

pub fn mp3<T>(len: usize) -> Vec<T>
    where T: Float + FloatConst + DCTnum
{
    let constant_term: T = T::PI() / T::from_usize(len).unwrap();
    let half: T = T::from_f32(0.5f32).unwrap();

    (0..len).map(|n| {
        let n_float: T = T::from_usize(n).unwrap();
        (constant_term * (n_float + half)).sin()
    }).collect()
}

pub fn vorbis<T>(len: usize) -> Vec<T>
    where T: Float + FloatConst + DCTnum
{
    let constant_term: T = T::PI() / T::from_usize(len).unwrap();
    let half: T = T::from_f32(0.5f32).unwrap();

    (0..len).map(|n| {
        let n_float: T = T::from_usize(n).unwrap();
        let inner_sin = (constant_term * (n_float + half)).sin();

        (T::FRAC_PI_2() * inner_sin * inner_sin).sin()
    }).collect()
}

pub fn one<T: DCTnum>(len: usize) -> Vec<T> {
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
                    assert!(fuzzy_cmp(first*first + second*second, 1f32, 0.001f32));
                }
            }
        }
    }
}