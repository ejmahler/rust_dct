use rustfft::{Planner};
use DCTnum;
use dct1::*;
use dct2::*;
use dct3::*;
use dct4::*;

pub struct DCTPlanner<T> {
	fft_planner: Planner<T>,
}
impl<T: DCTnum> DCTPlanner<T> {
	pub fn new() -> Self {
		Self {
			fft_planner: Planner::new(false)
		}
	}

	pub fn plan_dct1(&mut self, len: usize) -> Box<DCT1<T>> {
		//75 is a guess for the point at which converting to FFT will be faster than  performing the naive algorithm
		if len < 75 {
			Box::new(DCT1Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft((len - 1) * 2);
			Box::new(DCT1ViaFFT::new(fft))
		}
	}

	pub fn plan_dct2(&mut self, len: usize) -> Box<DCT2<T>> {
		//50 is a guess for the point at which converting to FFT will be faster than  performing the naive algorithm
		if len < 50 {
			Box::new(DCT2Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len);
			Box::new(DCT2ViaFFT::new(fft))
		}
	}

	pub fn plan_dct3(&mut self, len: usize) -> Box<DCT3<T>> {
		//50 is a guess for the point at which converting to FFT will be faster than  performing the naive algorithm
		if len < 50 {
			Box::new(DCT3Naive::new(len))
		} else {
			let fft = self.fft_planner.plan_fft(len);
			Box::new(DCT3ViaFFT::new(fft))
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
