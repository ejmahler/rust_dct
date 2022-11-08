use rustfft::num_traits::FloatConst;
use rustfft::FftNum;

/// Generic floating point number
pub trait DctNum: FftNum + FloatConst {
    fn half() -> Self;
    fn two() -> Self;
}

impl<T: FftNum + FloatConst> DctNum for T {
    fn half() -> Self {
        Self::from_f64(0.5).unwrap()
    }
    fn two() -> Self {
        Self::from_f64(2.0).unwrap()
    }
}

// Validates the given buffer verifying that it has the correct length.
macro_rules! validate_buffer {
    ($buffer: expr,$expected_buffer_len: expr) => {{
        if $buffer.len() != $expected_buffer_len {
            dct_error_inplace($buffer.len(), 0, $expected_buffer_len, 0);
            return;
        }
    }};
}

// Validates the given buffer and scratch by verifying that they have the correct length. Then, slices the scratch down to just the required amount
macro_rules! validate_buffers {
    ($buffer: expr, $scratch: expr, $expected_buffer_len: expr, $expected_scratch_len: expr) => {{
        if $buffer.len() != $expected_buffer_len {
            dct_error_inplace(
                $buffer.len(),
                $scratch.len(),
                $expected_buffer_len,
                $expected_scratch_len,
            );
            return;
        }
        if let Some(sliced_scratch) = $scratch.get_mut(0..$expected_scratch_len) {
            sliced_scratch
        } else {
            dct_error_inplace(
                $buffer.len(),
                $scratch.len(),
                $expected_buffer_len,
                $expected_scratch_len,
            );
            return;
        }
    }};
}

// Validates the given buffer and scratch by verifying that they have the correct length. Then, slices the scratch down to just the required amount
macro_rules! validate_buffers_mdct {
    ($buffer_a: expr, $buffer_b: expr, $buffer_c: expr, $scratch: expr, $expected_buffer_len: expr, $expected_scratch_len: expr) => {{
        if $buffer_a.len() != $expected_buffer_len
            || $buffer_b.len() != $expected_buffer_len
            || $buffer_c.len() != $expected_buffer_len
        {
            mdct_error_inplace(
                $buffer_a.len(),
                $buffer_b.len(),
                $buffer_c.len(),
                $scratch.len(),
                $expected_buffer_len,
                $expected_scratch_len,
            );
            return;
        }
        if let Some(sliced_scratch) = $scratch.get_mut(0..$expected_scratch_len) {
            sliced_scratch
        } else {
            mdct_error_inplace(
                $buffer_a.len(),
                $buffer_b.len(),
                $buffer_c.len(),
                $scratch.len(),
                $expected_buffer_len,
                $expected_scratch_len,
            );
            return;
        }
    }};
}

// Prints an error raised by an in-place FFT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
pub fn dct_error_inplace(
    actual_len: usize,
    actual_scratch: usize,
    expected_len: usize,
    expected_scratch: usize,
) {
    assert!(
        actual_len == expected_len,
        "Provided buffer must be equal to the transform size. Expected len = {}, got len = {}",
        expected_len,
        actual_len
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}

// Prints an error raised by an in-place FFT algorithm's `process_inplace` method
// Marked cold and inline never to keep all formatting code out of the many monomorphized process_inplace methods
#[cold]
#[inline(never)]
pub fn mdct_error_inplace(
    actual_len_a: usize,
    actual_len_b: usize,
    actual_len_c: usize,
    actual_scratch: usize,
    expected_len: usize,
    expected_scratch: usize,
) {
    assert!(
        actual_len_a == expected_len,
        "All three MDCT buffers must be equal to the transform size. Expected len = {}, but first buffer was len = {}",
        expected_len,
        actual_len_a
    );
    assert!(
        actual_len_b == expected_len,
        "All three  MDCT buffers must be equal to the transform size. Expected len = {}, but second buffer was len = {}",
        expected_len,
        actual_len_b
    );
    assert!(
        actual_len_c == expected_len,
        "All three  MDCT buffers must be equal to the transform size. Expected len = {}, but third buffer was len = {}",
        expected_len,
        actual_len_c
    );
    assert!(
        actual_scratch >= expected_scratch,
        "Not enough scratch space was provided. Expected scratch len >= {}, got scratch len = {}",
        expected_scratch,
        actual_scratch
    );
}
