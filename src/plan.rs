use rustfft::{Planner, FFTnum};
use dct4::{DCT4, DCT4ViaFFT};

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
		let fft = self.fft_planner.plan_fft(len * 8);
		Box::new(DCT4ViaFFT::new(fft))
	}
}
