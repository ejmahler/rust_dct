pub use rustfft;
pub use rustfft::num_complex;
pub use rustfft::num_traits;

use rustfft::Length;

#[macro_use]
mod common;

/// Algorithms for computing the Modified Discrete Cosine Transform
pub mod mdct;

pub mod algorithm;

mod array_utils;

mod plan;
mod twiddles;
pub use crate::common::DctNum;

pub use self::plan::DctPlanner;

#[cfg(test)]
mod test_utils;

pub trait RequiredScratch {
    fn get_scratch_len(&self) -> usize;
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 1 (DCT1)
pub trait Dct1<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 1 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct1_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct1(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct1_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 1 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 2 (DCT2)
pub trait Dct2<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 2 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct2_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct2(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct2_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 2 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 3 (DCT3)
pub trait Dct3<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 3 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct3_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct3(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct3_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 3 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 4 (DCT4)
pub trait Dct4<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 4 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst4_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct4(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct4_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 4 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 5 (DCT5)
pub trait Dct5<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 5 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct5_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct5(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct5_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 5 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct5_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 6 (DCT6)
pub trait Dct6<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 6 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct6_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct6(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct6_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 6 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct6_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 7 (DCT7)
pub trait Dct7<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 7 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct7_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct7(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct7_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 7 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct7_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 8 (DCT8)
pub trait Dct8<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DCT Type 8 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dct8_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dct8(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dct8_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DCT Type 8 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dct8_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Sine Transform Type 1 (DST1)
pub trait Dst1<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 1 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst1_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst1(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst1_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 1 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst1_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Sine Transform Type 2 (DST2)
pub trait Dst2<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 2 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst2_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst2(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst2_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 2 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst2_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Sine Transform Type 3 (DST3)
pub trait Dst3<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 3 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst3_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst3(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst3_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 3 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst3_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Sine Transform Type 4 (DST4)
pub trait Dst4<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 4 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst4_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst4(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst4_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 4 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst4_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 5 (DST5)
pub trait Dst5<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 5 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst4_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst5(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst5_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 5 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst5_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 6 (DST6)
pub trait Dst6<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 6 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst6_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst6(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst6_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 6 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst6_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 7 (DST7)
pub trait Dst7<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 7 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst7_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst7(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst7_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 7 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst7_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
}

/// A trait for algorithms which compute the Discrete Cosine Transform Type 8 (DST8)
pub trait Dst8<T: DctNum>: RequiredScratch + Length + Sync + Send {
    /// Computes the DST Type 8 on the provided buffer, in-place.
    ///
    /// This method may allocate a Vec<T> of scratch space as needed. If you'd like to reuse that allocation between
    /// multiple computations, consider calling `process_dst8_with_scratch` instead.
    ///
    /// Does not normalize outputs.
    fn process_dst8(&self, buffer: &mut [T]) {
        let mut scratch = vec![T::zero(); self.get_scratch_len()];
        self.process_dst8_with_scratch(buffer, &mut scratch);
    }
    /// Computes the DST Type 8 on the provided buffer, in-place. Uses the provided `scratch` buffer as scratch space.
    ///
    /// Does not normalize outputs.
    fn process_dst8_with_scratch(&self, buffer: &mut [T], scratch: &mut [T]);
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
