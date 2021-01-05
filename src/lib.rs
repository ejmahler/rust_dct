pub use rustfft;
pub use rustfft::num_complex;
pub use rustfft::num_traits;

use rustfft::Length;

/// Algorithms for computing the Modified Discrete Cosine Transform
pub mod mdct;

pub mod algorithm;

mod common;
mod plan;
mod twiddles;
pub use crate::common::DctNum;

pub use self::plan::DctPlanner;

#[cfg(test)]
mod test_utils;

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 1 (DCT1)
pub trait Dct1<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 1 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct1(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 2 (DCT2)
pub trait Dct2<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 2 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct2(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 3 (DCT3)
pub trait Dct3<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 3 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct3(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 4 (DCT4)
pub trait Dct4<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 4 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct4(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 5 (DCT5)
pub trait Dct5<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 5 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct5(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 6 (DCT6)
pub trait Dct6<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 6 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct6(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 7 (DCT7)
pub trait Dct7<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 7 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct7(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 8 (DCT8)
pub trait Dct8<T: DctNum>: Length + Sync + Send {
    /// Computes the DCT Type 8 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dct8(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 1 (DST1)
pub trait Dst1<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 1 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst1(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 2 (DST2)
pub trait Dst2<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 2 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst2(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 3 (DST3)
pub trait Dst3<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 3 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst3(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Sine Transform Type 4 (DST4)
pub trait Dst4<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 4 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst4(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 5 (DST5)
pub trait Dst5<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 5 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst5(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 6 (DST6)
pub trait Dst6<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 6 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst6(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 7 (DST7)
pub trait Dst7<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 7 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst7(&self, input: &mut [T], output: &mut [T]);
}

/// An umbrella trait for algorithms which compute the Discrete Cosine Transform Type 8 (DST8)
pub trait Dst8<T: DctNum>: Length + Sync + Send {
    /// Computes the DST Type 8 on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_dst8(&self, input: &mut [T], output: &mut [T]);
}

/// A trait for algorithms that can compute all of DCT2, DCT3, DST2, DST3, all in one struct
pub trait TransformType2And3<T: DctNum>: Dct2<T> + Dct3<T> + Dst2<T> + Dst3<T> {}

/// A trait for algorithms that can compute both DCT4 and DST4, all in one struct
pub trait TransformType4<T: DctNum>: Dct4<T> + Dst4<T> {}

/// A trait for algorithms that can compute both DCT6 and DCT7, all in one struct
pub trait Dct6And7<T: DctNum>: Dct6<T> + Dct7<T> {}

/// A trait for algorithms that can compute both DST6 and DST7, all in one struct
pub trait Dst6And7<T: DctNum>: Dst6<T> + Dst7<T> {}

#[test]
fn test_send_sync_impls() {
    fn assert_send_sync<T: ?Sized>()
    where
        T: Send + Sync,
    {
    }

    assert_send_sync::<dyn Dct1<f32>>();
    assert_send_sync::<dyn Dct2<f32>>();
    assert_send_sync::<dyn Dct3<f32>>();
    assert_send_sync::<dyn Dct4<f32>>();
    assert_send_sync::<dyn Dct5<f32>>();
    assert_send_sync::<dyn Dct6<f32>>();
    assert_send_sync::<dyn Dct7<f32>>();
    assert_send_sync::<dyn Dct8<f32>>();

    assert_send_sync::<dyn Dct1<f64>>();
    assert_send_sync::<dyn Dct2<f64>>();
    assert_send_sync::<dyn Dct3<f64>>();
    assert_send_sync::<dyn Dct4<f64>>();
    assert_send_sync::<dyn Dct5<f64>>();
    assert_send_sync::<dyn Dct6<f64>>();
    assert_send_sync::<dyn Dct7<f64>>();
    assert_send_sync::<dyn Dct8<f64>>();

    assert_send_sync::<dyn Dst1<f32>>();
    assert_send_sync::<dyn Dst2<f32>>();
    assert_send_sync::<dyn Dst3<f32>>();
    assert_send_sync::<dyn Dst4<f32>>();
    assert_send_sync::<dyn Dst5<f32>>();
    assert_send_sync::<dyn Dst6<f32>>();
    assert_send_sync::<dyn Dst7<f32>>();
    assert_send_sync::<dyn Dst8<f32>>();

    assert_send_sync::<dyn Dst1<f64>>();
    assert_send_sync::<dyn Dst2<f64>>();
    assert_send_sync::<dyn Dst3<f64>>();
    assert_send_sync::<dyn Dst4<f64>>();
    assert_send_sync::<dyn Dst5<f64>>();
    assert_send_sync::<dyn Dst6<f64>>();
    assert_send_sync::<dyn Dst7<f64>>();
    assert_send_sync::<dyn Dst8<f64>>();

    assert_send_sync::<dyn mdct::Mdct<f32>>();
    assert_send_sync::<dyn mdct::Mdct<f64>>();
}
