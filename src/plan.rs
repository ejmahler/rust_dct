use rustfft::{Planner, FFTnum};
use dct4::*;

pub struct DCTPlanner<T> {
	fft_planner: Planner<T>,
}
impl<T: FFTnum> DCTPlanner<T> {
	pub fn new() -> Self {
		Self {
			fft_planner: Planner::new(false)
		}
	}

	pub fn plan_dct4(&mut self, len: usize) -> Box<DCT4<T>> {
		//benchmarking show that below about 200, it's faster to just use the naive DCT4 algorithm
		if len < 200 {
			Box::new(DCT4Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len * 8);
			Box::new(DCT4ViaFFT::new(fft))
		}
	}
}
