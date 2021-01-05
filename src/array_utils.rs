use rustfft::num_complex::Complex;

#[allow(unused)]
pub fn into_complex<T>(buffer: &[T]) -> &[Complex<T>] {
    let complex_len = buffer.len() / 2;
    let ptr = buffer.as_ptr() as *const Complex<T>;
    unsafe { std::slice::from_raw_parts(ptr, complex_len) }
}

pub fn into_complex_mut<T>(buffer: &mut [T]) -> &mut [Complex<T>] {
    let complex_len = buffer.len() / 2;
    let ptr = buffer.as_mut_ptr() as *mut Complex<T>;
    unsafe { std::slice::from_raw_parts_mut(ptr, complex_len) }
}
