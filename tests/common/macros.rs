use crate::common::{compare_float_vectors, random_signal};
use rustdct::DctPlanner;

macro_rules! dct_test_with_known_data {
    ($reference_fn:ident, $naive_struct:ident, $process_fn: ident, $known_data_fn:ident) => (
        // Compare our naive struct and our reference_fn implementation against a bunch of known data
        let known_data = $known_data_fn();
        for entry in known_data {
            let len = entry.input.len();
            assert_eq!(len, entry.expected_output.len(), "Invalid test data -- input and known output are not the same length");

            let mut naive_buffer = entry.input.clone();

            let naive_dct = $naive_struct::new(len);
            naive_dct.$process_fn(&mut naive_buffer);

            let slow_output = $reference_fn(&entry.input);

            println!("input:          {:?}", entry.input);
            println!("expected output:{:?}", entry.expected_output);
            println!("naive output:   {:?}", naive_buffer);
            println!("slow output:    {:?}", slow_output);

            assert!(compare_float_vectors(&entry.expected_output, &naive_buffer));
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

            let mut naive_buffer = input.clone();
            let mut actual_buffer = input.clone();

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
            naive_dct.$process_fn(&mut naive_buffer);
            actual_dct.$process_fn(&mut actual_buffer);

            println!("input:           {:?}", input);
            println!("reference output:{:?}", reference_output);
            println!("naive output:    {:?}", naive_buffer);
            println!("planned output:  {:?}", actual_buffer);

            assert!(compare_float_vectors(&reference_output, &naive_buffer));
            assert!(compare_float_vectors(&reference_output, &actual_buffer));
        }
    };
}

pub mod test_mdct {
    use super::*;
    use rustdct::{
        mdct::{Mdct, MdctNaive},
        RequiredScratch,
    };

    pub fn planned_matches_naive<F>(len: usize, window_fn: F)
    where
        F: Fn(usize) -> Vec<f32>,
    {
        let input = random_signal(len * 2);
        println!("input:          {:?}", input);

        let (input_a, input_b) = input.split_at(len);

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

        let mut naive_scratch = vec![0f32; naive_dct.get_scratch_len()];
        let mut fast_scratch = vec![0f32; actual_dct.get_scratch_len()];

        naive_dct.process_mdct_with_scratch(
            input_a,
            input_b,
            &mut naive_output,
            &mut naive_scratch,
        );
        actual_dct.process_mdct_with_scratch(
            input_a,
            input_b,
            &mut actual_output,
            &mut fast_scratch,
        );

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
        let mut scratch = vec![0f32; mdct.get_scratch_len()];

        for i in 0..NUM_SEGMENTS {
            let input_chunk = &input[len * i..(len * (i + 2))];
            let output_chunk = &mut output[len * i..(len * (i + 1))];

            let (input_a, input_b) = input_chunk.split_at(len);

            mdct.process_mdct_with_scratch(input_a, input_b, output_chunk, &mut scratch);
        }
        for i in 0..NUM_SEGMENTS {
            let input_chunk = &output[len * i..(len * (i + 1))];
            let output_chunk = &mut inverse[len * i..(len * (i + 2))];

            let (output_a, output_b) = output_chunk.split_at_mut(len);

            mdct.process_imdct_with_scratch(input_chunk, output_a, output_b, &mut scratch);
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
