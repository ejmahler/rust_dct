use std::f64;

/// Simplified version of DCT1 that doesn't precompute twiddles. slower but much easier to debug
pub fn slow_dct1(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(input.len());
    let twiddle_constant = f64::consts::PI / ((input.len() - 1) as f64);

    for k in 0..input.len() {
        let mut current_value = if k % 2 == 0 {
            (input[0] + input[input.len() - 1]) * 0.5
        } else {
            (input[0] - input[input.len() - 1]) * 0.5
        };

        let k_float = k as f64;

        for i in 1..(input.len() - 1) {
            let i_float = i as f64;

            let twiddle = (k_float * i_float * twiddle_constant).cos();

            current_value += input[i] * twiddle;
        }
        result.push(current_value);

    }
    return result;
}

/// Simplified version of DCT2 that doesn't precompute twiddles. slower but much easier to debug
pub fn slow_dct2(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(input.len());
    let size_float = input.len() as f64;

    for k in 0..input.len() {
        let mut current_value = 0.0;

        let k_float = k as f64;

        for i in 0..(input.len()) {
            let i_float = i as f64;

            let twiddle = (f64::consts::PI * k_float * (i_float + 0.5) / size_float).cos();

            current_value += input[i] * twiddle;
        }
        result.push(current_value);

    }
    return result;
}

/// Simplified version of DCT3 that doesn't precompute twiddles. slower but much easier to debug
pub fn slow_dct3(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(input.len());

    let size_float = input.len() as f64;

    for k in 0..input.len() {
        let mut current_value = input[0] * 0.5;

        let k_float = k as f64;

        for i in 1..(input.len()) {
            let i_float = i as f64;

            let twiddle = (f64::consts::PI * i_float * (k_float + 0.5) / size_float).cos();

            current_value += input[i] * twiddle;
        }
        result.push(current_value);

    }

    return result;
}

/// Simplified version of DCT4 that doesn't precompute twiddles. slower but much easier to debug
pub fn slow_dct4(input: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(input.len());

    let size_float = input.len() as f64;


    for k in 0..input.len() {
        let mut current_value = 0.0;

        let k_float = k as f64;

        for i in 0..input.len() {
            let i_float = i as f64;

            current_value += input[i] *
                (f64::consts::PI * (i_float + 0.5) * (k_float + 0.5) / size_float)
                    .cos();
        }
        result.push(current_value);

    }

    return result;
}