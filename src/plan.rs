use rustfft::FFTplanner;
use DCTnum;
use dct1::*;
use dct2::*;
use dct3::*;
use dct4::*;
use mdct::*;

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
}
impl<T: DCTnum> DCTplanner<T> {
    pub fn new() -> Self {
        Self { fft_planner: FFTplanner::new(false) }
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
    ///If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct2(&mut self, len: usize) -> Box<DCT2<T>> {
        //benchmarking shows that below about 5, it's faster to just use the naive DCT1 algorithm
        if len < 5 {
            Box::new(DCT2Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft(len);
            Box::new(DCT2ViaFFT::new(fft))
        }
    }

    /// Returns a DCT Type 3 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct3(&mut self, len: usize) -> Box<DCT3<T>> {
        //benchmarking shows that below about 5, it's faster to just use the naive DCT1 algorithm
        if len < 5 {
            Box::new(DCT3Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft(len);
            Box::new(DCT3ViaFFT::new(fft))
        }
    }

    /// Returns a DCT Type 4 instance which processes signals of size `len`.
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_dct4(&mut self, len: usize) -> Box<DCT4<T>> {
        //benchmarking shows that below about 100, it's faster to just use the naive DCT4 algorithm
        if len < 100 {
            Box::new(DCT4Naive::new(len))
        } else {
            let fft = self.fft_planner.plan_fft(len * 4);
            Box::new(DCT4ViaFFT::new(fft))
        }
    }

    /// Returns a MDCT instance which processes inputs of size ` len * 2` and produces outputs of size `len`.
    ///
    /// `window_fn` is a function that takes a `size` and returns a `Vec` containing `size` window values.
    /// See the [`window_fn`](mdct/window_fn/index.html) module for provided window functions.
    ///
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_mdct<F>(&mut self, len: usize, window_fn: F) -> Box<MDCT<T>>
        where F: Fn(usize) -> Vec<T>
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
        where F: Fn(usize) -> Vec<T>
    {
        //benchmarking shows that using the inner dct4 algorithm is always faster than computing the naive algorithm
        let inner_dct4 = self.plan_dct4(len);
        Box::new(IMDCTViaDCT4::new(inner_dct4, window_fn))
    }
}
