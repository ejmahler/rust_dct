use std::f64;

// This file contains "slow" versions of all DCT and DST transforms.
// The goal of these implementations is not to be fast, but to match the mathematical definitions as closely as possible and to be easy to follow and debug
// The reference for the mathematical definitions was section 9 of "The Discrete W Transforms" by Wang and Hunt, but with the normalization/orthogonalization factors omitted.

/// Simplified version of DCT1
pub fn slow_dct1(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 || input_index == input.len() - 1 { 0.5 } else { 1.0 };
            let cos_inner = (output_index as f64) * (input_index as f64) * f64::consts::PI / ((input.len() - 1) as f64);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DCT2
pub fn slow_dct2(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let cos_inner = (output_index as f64) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT3
pub fn slow_dct3(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
            let cos_inner = (output_index as f64 + 0.5) * (input_index as f64) * f64::consts::PI / (input.len() as f64);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT4
pub fn slow_dct4(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let cos_inner = (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT5
pub fn slow_dct5(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
            let cos_inner = (output_index as f64) * (input_index as f64) * f64::consts::PI / (input.len() as f64 - 0.5);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT6
pub fn slow_dct6(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == input.len() - 1 { 0.5 } else { 1.0 };
            let cos_inner = (output_index as f64) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64 - 0.5);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT7
pub fn slow_dct7(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
            let cos_inner = (output_index as f64 + 0.5) * (input_index as f64) * f64::consts::PI / (input.len() as f64 - 0.5);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DCT8
pub fn slow_dct8(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let cos_inner = (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64 + 0.5);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}


/// Simplified version of DST1
pub fn slow_dst1(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 1.0) * (input_index as f64 + 1.0) * f64::consts::PI / ((input.len() + 1) as f64);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST2
pub fn slow_dst2(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 1.0) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST3
pub fn slow_dst3(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == input.len() - 1 { 0.5 } else { 1.0 };
            let sin_inner = (output_index as f64 + 0.5) * (input_index as f64 + 1.0) * f64::consts::PI / (input.len() as f64);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST4
pub fn slow_dst4(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}

/// Simplified version of DST5
pub fn slow_dst5(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 1.0) * (input_index as f64 + 1.0) * f64::consts::PI / ((input.len()) as f64 + 0.5);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST6
pub fn slow_dst6(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 1.0) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64 + 0.5);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST7
pub fn slow_dst7(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let sin_inner = (output_index as f64 + 0.5) * (input_index as f64 + 1.0) * f64::consts::PI / (input.len() as f64 + 0.5);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }
    result
}

/// Simplified version of DST8
pub fn slow_dst8(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == input.len() - 1 { 0.5 } else { 1.0 };
            let sin_inner = (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * f64::consts::PI / (input.len() as f64 - 0.5);
            let twiddle = sin_inner.sin();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}
