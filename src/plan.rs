use std::collections::HashMap;
use std::sync::Arc;

use rustfft::FFTplanner;
use DCTnum;
use dct1::*;
use dct2::*;
use dct2::dct2_butterflies::*;
use dct3::*;
use dct4::*;
use mdct::*;

const DCT2_BUTTERFLIES: [usize; 4] = [2, 4, 8, 16];

/// The DCT planner is used to make new DCT algorithm instances.
///
/// RustDCT has several DCT algorithms available for each DCT type; For a given DCT size and type, the FFTplanner
/// decides which of the available DCT algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a DCT Type 4 of size 1234
/// use rustdct::DCTplanner;
///
/// let mut input:  Vec<f32> = vec![0f32; 1234];
/// let mut output: Vec<f32> = vec![0f32; 1234];
///
/// let mut planner = DCTplanner::new();
/// let mut dct4 = planner.plan_dct4(1234);
/// dct4.process(&mut input, &mut output);
/// ~~~
///
/// If you plan on creating multiple DCT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across DCT instances wherever possible, saving memory and reducing
/// setup time. (DCT instances created with one planner will never re-use data and buffers with DCT instances created
/// by a different planner)
///
/// Each FFT instance owns `Arc`s to its shared internal data, rather than borrowing it from the planner, so it's
/// perfectly safe to drop the planner after creating DCT instances.
pub struct DCTplanner<T> {
    fft_planner: FFTplanner<T>,
    dct2_cache: HashMap<usize, Arc<DCT2<T>>>,
}
impl<T: DCTnum> DCTplanner<T> {
    pub fn new() -> Self {
        Self {
            fft_planner: FFTplanner::new(false),
            dct2_cache: HashMap::new(),
        }
    }

    /// Returns a DCT Type 1 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct1(&mut self, len: usize) -> Box<DCT1<T>> {
        //benchmarking shows that below about 25, it's faster to just use the naive DCT1 algorithm
        if len < 25 {
            Box::new(DCT1Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft((len - 1) * 2);
            Box::new(DCT1ViaFFT::new(fft))
        }
    }




    /// Returns a DCT Type 2 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct2(&mut self, len: usize) -> Arc<DCT2<T>> {
        if self.dct2_cache.contains_key(&len) {
            self.dct2_cache.get(&len).unwrap().clone()
        } else {
            let result = self.plan_new_dct2(len);
            self.dct2_cache.insert(len, result.clone());
            result
        }
    }

    fn plan_new_dct2(&mut self, len: usize) -> Arc<DCT2<T>> {
        if DCT2_BUTTERFLIES.contains(&len) {
            self.plan_dct2_butterfly(len)
        } else if len.is_power_of_two() && len > 2 {
            let half_dct = self.plan_dct2(len / 2);
            let quarter_dct = self.plan_dct2(len / 4);
            Arc::new(DCT2SplitRadix::new(half_dct, quarter_dct)) as Arc<DCT2<T>>
        } else if len < 16 {
            //benchmarking shows that below about 16, it's faster to just use the naive DCT2 algorithm
            Arc::new(DCT2Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft(len);
            Arc::new(DCT2ViaFFT::new(fft))
        }
    }

    fn plan_dct2_butterfly(&mut self, len: usize) -> Arc<DCT2<T>> {
        match len {
            2 => Arc::new(DCT2Butterfly2::new()),
            4 => Arc::new(DCT2Butterfly4::new()),
            8 => Arc::new(DCT2Butterfly8::new()),
            16 => Arc::new(DCT2Butterfly16::new()),
            _ => panic!("Invalid butterfly size for DCT2: {}", len)
        }
    }




    /// Returns a DCT Type 3 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct3(&mut self, len: usize) -> Box<DCT3<T>> {
        if len.is_power_of_two() && len > 2 {
            Box::new(DCT3SplitRadix::new(len))
        } else if len < 5 {
            //benchmarking shows that below about 5, it's faster to just use the naive DCT3 algorithm
            Box::new(DCT3Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft(len);
            Box::new(DCT3ViaFFT::new(fft))
        }
    }

    /// Returns a DCT Type 4 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct4(&mut self, len: usize) -> Box<DCT4<T>> {

        //if we have an even size, we can use the DCT4 Via DCT3 algorithm
        if len % 2 == 0 {
            //benchmarking shows that below 6, it's faster to just use the naive DCT4 algorithm
            if len < 6 {
                Box::new(DCT4Naive::new(len))
            } else {
                let inner_dct = self.plan_dct3(len / 2);
                Box::new(DCT4ViaDCT3::new(inner_dct))
            }

        }
        else {
            //odd size, so we can use the "DCT4 via FFT odd" algorithm
            //benchmarking shows that below about 7, it's faster to just use the naive DCT4 algorithm
            if len < 7 {
                Box::new(DCT4Naive::new(len))
            } else {
                let fft = self.fft_planner.plan_fft(len);
                Box::new(DCT4ViaFFTOdd::new(fft))
            }
        }
    }

    /// Returns a MDCT instance which processes inputs of size ` len * 2` and produces outputs of size `len`.
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    ///
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_mdct<F>(&mut self, len: usize, window_fn: F) -> Box<MDCT<T>>
    where
        F: FnOnce(usize) -> Vec<T>,
    {
        //benchmarking shows that using the inner dct4 algorithm is always faster than computing the naive algorithm
        let inner_dct4 = self.plan_dct4(len);
        Box::new(MDCTViaDCT4::new(inner_dct4, window_fn))
    }

    /// Returns an IMDCT instance which processes input of size `len` and produces outputs of size `len * 2`.
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    ///
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_imdct<F>(&mut self, len: usize, window_fn: F) -> Box<IMDCT<T>>
    where
        F: FnOnce(usize) -> Vec<T>,
    {
        //benchmarking shows that using the inner dct4 algorithm is always faster than computing the naive algorithm
        let inner_dct4 = self.plan_dct4(len);
        Box::new(IMDCTViaDCT4::new(inner_dct4, window_fn))
    }
}
