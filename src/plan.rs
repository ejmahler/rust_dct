use rustfft::FFTplanner;
use DCTnum;
use dct1::*;
use dct2::*;
use dct3::*;
use dct4::*;
use mdct::*;

pub struct DCTPlanner<T> {
	fft_planner: FFTplanner<T>,
}
impl<T: DCTnum> DCTPlanner<T> {
	pub fn new() -> Self {
		Self {
			fft_planner: FFTplanner::new(false)
		}
	}

	pub fn plan_dct1(&mut self, len: usize) -> Box<DCT1<T>> {
		//benchmarking shows that below about 25, it's faster to just use the naive DCT1 algorithm
		if len < 25 {
			Box::new(DCT1Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft((len - 1) * 2);
			Box::new(DCT1ViaFFT::new(fft))
		}
	}

	pub fn plan_dct2(&mut self, len: usize) -> Box<DCT2<T>> {
		//benchmarking shows that below about 5, it's faster to just use the naive DCT1 algorithm
		if len < 5 {
			Box::new(DCT2Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len);
			Box::new(DCT2ViaFFT::new(fft))
		}
	}

	pub fn plan_dct3(&mut self, len: usize) -> Box<DCT3<T>> {
		//benchmarking shows that below about 5, it's faster to just use the naive DCT1 algorithm
		if len < 5 {
			Box::new(DCT3Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len);
			Box::new(DCT3ViaFFT::new(fft))
		}
	}

	pub fn plan_dct4(&mut self, len: usize) -> Box<DCT4<T>> {
		//benchmarking shows that below about 100, it's faster to just use the naive DCT4 algorithm
		if len < 100 {
			Box::new(DCT4Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len * 4);
			Box::new(DCT4ViaFFT::new(fft))
		}
	}

	pub fn plan_mdct<F>(&mut self, len: usize, window_fn: F) -> Box<MDCT<T>>
		where F: Fn(usize) -> Vec<T>
	{
		//benchmarking shows that using the inner dct4 algorithm is always faster than computing the naive algorithm
		let inner_dct4 = self.plan_dct4(len);
		Box::new(MDCTViaDCT4::new(inner_dct4, window_fn))
	}

	pub fn plan_imdct<F>(&mut self, len: usize, window_fn: F) -> Box<IMDCT<T>>
		where F: Fn(usize) -> Vec<T>
	{
		//benchmarking shows that using the inner dct4 algorithm is always faster than computing the naive algorithm
		let inner_dct4 = self.plan_dct4(len);
		Box::new(IMDCTViaDCT4::new(inner_dct4, window_fn))
	}
}
