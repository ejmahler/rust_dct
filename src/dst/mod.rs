use rustfft::Length;
use common;

mod dst1_naive;
mod dst2_naive;
mod dst3_naive;
mod dst4_naive;

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 1 (DST1)
pub trait DST1<T: common::DCTnum>: Length {
    /// Computes the DST Type 1 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 2 (DST2)
pub trait DST2<T: common::DCTnum>: Length {
    /// Computes the DST Type 2 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 3 (DST3)
pub trait DST3<T: common::DCTnum>: Length {
    /// Computes the DST Type 3 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 4 (DST4)
pub trait DST4<T: common::DCTnum>: Length {
    /// Computes the DST Type 4 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [T], output: &mut [T]);
}

pub use self::dst1_naive::DST1Naive;
pub use self::dst2_naive::DST2Naive;
pub use self::dst3_naive::DST3Naive;
pub use self::dst4_naive::DST4Naive;
