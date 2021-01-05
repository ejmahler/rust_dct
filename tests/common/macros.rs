use crate::common::{compare_float_vectors, random_signal};
use rustdct::DctPlanner;

macro_rules! dct_test_with_known_data {
    ($reference_fn:ident, $naive_struct:ident, $process_fn: ident, $known_data_fn:ident) => (
        // Compare our naive struct and our reference_fn implementation against a bunch of known data
        let known_data = $known_data_fn();
        for entry in known_data {
            let len = entry.input.len();
            assert_eq!(len, entry.expected_output.len(), "Invalid test data -- input and known output are not the same length");

            let mut naive_input = entry.input.clone();
            let mut naive_output = vec![0.0; len];

            let naive_dct = $naive_struct::new(len);
            naive_dct.$process_fn(&mut naive_input, &mut naive_output);

            let slow_output = $reference_fn(&entry.input);

            println!("input:          {:?}", entry.input);
            println!("expected output:{:?}", entry.expected_output);
            println!("naive output:   {:?}", naive_output);
            println!("slow output:    {:?}", slow_output);

            assert!(compare_float_vectors(&entry.expected_output, &naive_output));
            assert!(compare_float_vectors(&entry.expected_output, &slow_output));
        }
    )
}

macro_rules! dct_test_inverse {
    ($reference_fn:ident, $inverse_fn:ident, $inverse_scale_fn:ident, $first_size:expr) => (
        // Test that the slow fn, paired with the correct inverse fn, actually yields the original data
        for len in $first_size..20 {
            let input = random_signal(len);
            let intermediate = $reference_fn(&input);
            let inverse = $inverse_fn(&intermediate);

            let inverse_scale = $inverse_scale_fn(len);
            let scaled_inverse: Vec<f64> = inverse.into_iter().map(|entry| entry * inverse_scale).collect();

            println!("input:          {:?}", input);
            println!("scaled inverse: {:?}", scaled_inverse);

            assert!(compare_float_vectors(&input, &scaled_inverse));
        }
    )
}

macro_rules! dct_test_with_planner {
    ($reference_fn:ident, $naive_struct:ident, $process_fn: ident, $planner_fn:ident, $first_size:expr) => {
        // Compare our naive struct against the output from the planner
        for len in $first_size..20 {
            let input = random_signal(len);

            let mut naive_input = input.clone();
            let mut actual_input = input.clone();

            let mut naive_output = vec![0.0; len];
            let mut actual_output = vec![0.0; len];

            let naive_dct = $naive_struct::new(len);

            let mut planner = DctPlanner::new();
            let actual_dct = planner.$planner_fn(len);

            assert_eq!(
                actual_dct.len(),
                len,
                "Planner created a DCT of incorrect length. Expected {}, got {}",
                len,
                actual_dct.len()
            );

            let reference_output = $reference_fn(&input);
            naive_dct.$process_fn(&mut naive_input, &mut naive_output);
            actual_dct.$process_fn(&mut actual_input, &mut actual_output);

            println!("input:           {:?}", input);
            println!("reference output:{:?}", reference_output);
            println!("naive output:    {:?}", naive_output);
            println!("planned output:  {:?}", actual_output);

            assert!(compare_float_vectors(&reference_output, &naive_output));
            assert!(compare_float_vectors(&reference_output, &actual_output));
        }
    };
}

pub mod test_mdct {
    use super::*;
    use rustdct::mdct::{Mdct, MdctNaive};

    pub fn planned_matches_naive<F>(len: usize, window_fn: F)
    where
        F: Fn(usize) -> Vec<f32>,
    {
        let mut naive_input = random_signal(len * 2);
        let mut actual_input = naive_input.clone();

        println!("input:          {:?}", naive_input);

        let mut naive_output = vec![0f32; len];
        let mut actual_output = vec![0f32; len];

        let naive_dct = MdctNaive::new(len, &window_fn);

        let mut planner = DctPlanner::new();
        let actual_dct = planner.plan_mdct(len, window_fn);

        assert_eq!(
            actual_dct.len(),
            len,
            "Planner created a DCT of incorrect length"
        );

        naive_dct.process_mdct(&mut naive_input, &mut naive_output);
        actual_dct.process_mdct(&mut actual_input, &mut actual_output);

        println!("Naive output:   {:?}", naive_output);
        println!("Planned output: {:?}", actual_output);

        assert!(
            compare_float_vectors(&naive_output, &actual_output),
            "len = {}",
            len
        );
    }

    pub fn test_tdac<F>(len: usize, scale_factor: f32, window_fn: F)
    where
        F: Fn(usize) -> Vec<f32>,
    {
        let mut planner = DctPlanner::new();
        let mdct = planner.plan_mdct(len, &window_fn);

        const NUM_SEGMENTS: usize = 5;

        let input = random_signal(len * (NUM_SEGMENTS + 1));
        let mut output = vec![0f32; len * NUM_SEGMENTS];
        let mut inverse = vec![0f32; len * (NUM_SEGMENTS + 1)];

        for i in 0..NUM_SEGMENTS {
            let input_chunk = &input[len * i..(len * (i + 2))];
            let output_chunk = &mut output[len * i..(len * (i + 1))];

            mdct.process_mdct(input_chunk, output_chunk);
        }
        for i in 0..NUM_SEGMENTS {
            let input_chunk = &output[len * i..(len * (i + 1))];
            let output_chunk = &mut inverse[len * i..(len * (i + 2))];

            mdct.process_imdct(input_chunk, output_chunk);
        }

        //we have to scale the inverse by 1/len
        for element in inverse.iter_mut() {
            *element = *element * scale_factor;
        }

        println!("scale:   {:?}", scale_factor);
        println!("input:   {:?}", &input[len..input.len() - len]);
        println!("inverse: {:?}", &inverse[len..input.len() - len]);

        assert!(
            compare_float_vectors(
                &input[len..input.len() - len],
                &inverse[len..inverse.len() - len],
            ),
            "len = {}",
            len
        );
    }
}
