use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};

#[allow(unused_imports)]
use rustdct::DCTplanner;
#[allow(unused_imports)]
use rustdct::dct1::{DCT1, DCT1Naive};
#[allow(unused_imports)]
use rustdct::dct2::{DCT2, DCT2Naive};
#[allow(unused_imports)]
use rustdct::dct3::{DCT3, DCT3Naive};
#[allow(unused_imports)]
use rustdct::dct4::{DCT4, DCT4Naive};

pub fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
    a >= b - tolerance && a <= b + tolerance
}

pub fn compare_float_vectors(expected: &[f32], observed: &[f32]) -> bool {
    assert_eq!(expected.len(), observed.len());

    let tolerance: f32 = 0.001;

    for i in 0..expected.len() {
        if !fuzzy_cmp(observed[i], expected[i], tolerance) {
            return false;
        }
    }
    true
}

pub fn random_signal(length: usize) -> Vec<f32> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(0.0, 10.0);

    let seed: [usize; 5] = [1910, 11431, 4984, 14828, length];
    let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

    for _ in 0..length {
        sig.push(normal_dist.ind_sample(&mut rng) as f32);
    }
    return sig;
}

macro_rules! dct_test_fns {
    ($module_name:ident, $naive_struct:ident, $planner_fn:ident) => (
        pub mod $module_name {
            use super::*;

            pub fn compare_known_length(input: &[f32], known_output: &[f32]) {
                assert_eq!(input.len(), known_output.len(), "Input and known output must have the same length");

                let len = input.len();

                let mut naive_input = input.to_vec();
                let mut actual_input = input.to_vec();

                let mut naive_output = vec![0f32; len];
                let mut actual_output = vec![0f32; len];

                let mut naive_dct = $naive_struct::new(len);

                let mut planner = DCTplanner::new();
                let mut actual_dct = planner.$planner_fn(len);

                assert_eq!(actual_dct.len(), len, "Planner created a DCT of incorrect length");


                naive_dct.process(&mut naive_input, &mut naive_output);
                actual_dct.process(&mut actual_input, &mut actual_output);

                println!("input:          {:?}", input);
                println!("known output:   {:?}", known_output);
                println!("Naive output:   {:?}", naive_output);
                println!("Planned output: {:?}", actual_output);

                assert!(compare_float_vectors(&known_output, &naive_output));
                assert!(compare_float_vectors(&known_output, &actual_output));
            }

            pub fn planned_matches_naive(len: usize) {
                let mut naive_input = random_signal(len);
                let mut actual_input = naive_input.clone();

                println!("input:          {:?}", naive_input);

                let mut naive_output = vec![0f32; len];
                let mut actual_output = vec![0f32; len];

                let mut naive_dct = $naive_struct::new(len);

                let mut planner = DCTplanner::new();
                let mut actual_dct = planner.$planner_fn(len);

                assert_eq!(actual_dct.len(), len, "Planner created a DCT of incorrect length");

                naive_dct.process(&mut naive_input, &mut naive_output);
                actual_dct.process(&mut actual_input, &mut actual_output);

                println!("Naive output:   {:?}", naive_output);
                println!("Planned output: {:?}", actual_output);

                assert!(compare_float_vectors(&naive_output, &actual_output), "len = {}", len);
            }
        }
    )
}
dct_test_fns!(test_dct1, DCT1Naive, plan_dct1);
dct_test_fns!(test_dct2, DCT2Naive, plan_dct2);
dct_test_fns!(test_dct3, DCT3Naive, plan_dct3);
dct_test_fns!(test_dct4, DCT4Naive, plan_dct4);


pub mod test_mdct {
    use super::*;
    use rustdct::mdct::{MDCT, MDCTNaive};

    pub fn planned_matches_naive<F>(len: usize, window_fn: F)
        where F: Fn(usize) -> Vec<f32>
    {
        let mut naive_input = random_signal(len * 2);
        let mut actual_input = naive_input.clone();

        println!("input:          {:?}", naive_input);

        let mut naive_output = vec![0f32; len];
        let mut actual_output = vec![0f32; len];

        let mut naive_dct = MDCTNaive::new(len, &window_fn);

        let mut planner = DCTplanner::new();
        let mut actual_dct = planner.plan_mdct(len, window_fn);

        assert_eq!(actual_dct.len(),
                   len,
                   "Planner created a DCT of incorrect length");

        naive_dct.process(&mut naive_input, &mut naive_output);
        actual_dct.process(&mut actual_input, &mut actual_output);

        println!("Naive output:   {:?}", naive_output);
        println!("Planned output: {:?}", actual_output);

        assert!(compare_float_vectors(&naive_output, &actual_output),
                "len = {}",
                len);
    }

    pub fn test_tdac<F>(len: usize, scale_factor: f32, window_fn: F)
        where F: Fn(usize) -> Vec<f32>
    {
        let mut planner = DCTplanner::new();
        let mut forward_dct = planner.plan_mdct(len, &window_fn);
        let mut inverse_dct = planner.plan_imdct(len, window_fn);

        const NUM_SEGMENTS: usize = 5;

        let input = random_signal(len * (NUM_SEGMENTS + 1));
        let mut output = vec![0f32; len * NUM_SEGMENTS];
        let mut inverse = vec![0f32; len * (NUM_SEGMENTS + 1)];

        for i in 0..NUM_SEGMENTS {
            let input_chunk = &input[len * i..(len * (i + 2))];
            let output_chunk = &mut output[len * i..(len * (i + 1))];

            forward_dct.process(input_chunk, output_chunk);
        }
        for i in 0..NUM_SEGMENTS {
            let input_chunk = &output[len * i..(len * (i + 1))];
            let output_chunk = &mut inverse[len * i..(len * (i + 2))];

            inverse_dct.process(input_chunk, output_chunk);
        }

        //we have to scale the inverse by 1/len
        for element in inverse.iter_mut() {
            *element = *element * scale_factor;
        }

        assert!(compare_float_vectors(&input[len..input.len() - len],
                                      &inverse[len..inverse.len() - len]),
                "len = {}",
                len);
    }
}

pub mod test_imdct {
    use super::*;
    use rustdct::mdct::{IMDCT, IMDCTNaive};

    pub fn planned_matches_naive<F>(len: usize, window_fn: F)
        where F: Fn(usize) -> Vec<f32>
    {
        let mut naive_input = random_signal(len);
        let mut actual_input = naive_input.clone();

        println!("input:          {:?}", naive_input);

        let mut naive_output = vec![0f32; len*2];
        let mut actual_output = vec![0f32; len*2];

        let mut naive_dct = IMDCTNaive::new(len, &window_fn);

        let mut planner = DCTplanner::new();
        let mut actual_dct = planner.plan_imdct(len, window_fn);

        assert_eq!(actual_dct.len(),
                   len,
                   "Planner created a DCT of incorrect length");

        naive_dct.process(&mut naive_input, &mut naive_output);
        actual_dct.process(&mut actual_input, &mut actual_output);

        println!("Naive output:   {:?}", naive_output);
        println!("Planned output: {:?}", actual_output);

        assert!(compare_float_vectors(&naive_output, &actual_output),
                "len = {}",
                len);
    }
}
