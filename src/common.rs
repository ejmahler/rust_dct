use rustfft::num_traits::FloatConst;
use rustfft::FftNum;

/// Generic floating point number
pub trait DctNum: FftNum + FloatConst {
    fn half() -> Self;
    fn two() -> Self;
}

impl<T: FftNum + FloatConst> DctNum for T {
    fn half() -> Self {
        Self::from_f64(0.5).unwrap()
    }
    fn two() -> Self {
        Self::from_f64(2.0).unwrap()
    }
}

#[inline(always)]
pub fn verify_length<T>(input: &[T], output: &[T], expected: usize) {
    assert_eq!(
        input.len(),
        expected,
        "Input is the wrong length. Expected {}, got {}",
        expected,
        input.len()
    );
    assert_eq!(
        output.len(),
        expected,
        "Output is the wrong length. Expected {}, got {}",
        expected,
        output.len()
    );
}

#[allow(unused)]
#[inline(always)]
pub fn verify_length_divisible<T>(input: &[T], output: &[T], expected: usize) {
    assert_eq!(
        input.len() % expected,
        0,
        "Input is the wrong length. Expected multiple of {}, got {}",
        expected,
        input.len()
    );
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output must have the same length. Expected {}, got {}",
        input.len(),
        output.len()
    );
}
