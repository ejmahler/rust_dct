
pub fn transpose<T>(width: usize, height: usize, input: &[T], output: &mut [T])
    where T: Copy
{
    assert_eq!(width * height, input.len());

    for y in 0..height {
        for x in 0..width {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use std::f32;

    fn fuzzy_cmp(a: f32, b: f32, tolerance: f32) -> bool {
        a >= b - tolerance && a <= b + tolerance
    }

    fn compare_float_vectors(expected: &[f32], observed: &[f32]) {
        assert_eq!(expected.len(), observed.len());

        let tolerance: f32 = 0.0001;

        for i in 0..expected.len() {
            assert!(fuzzy_cmp(observed[i], expected[i], tolerance));
        }
    }

    #[test]
    fn test_transpose() {
        let input_list = vec![
            (2 as usize, 2 as usize, vec![
                1_f32, 2_f32,
                3_f32, 4_f32,
            ]),
            (3 as usize, 2 as usize, vec![
                1_f32, 2_f32, 3_f32,
                4_f32, 5_f32, 6_f32
            ]),
            (2 as usize, 3 as usize, vec![
                1_f32, 2_f32,
                3_f32, 4_f32,
                5_f32, 6_f32,

            ]),
        ];
        let expected_list = vec![
            vec![
                1_f32, 3_f32,
                2_f32, 4_f32,
            ],
            vec![
                1_f32, 4_f32,
                2_f32, 5_f32,
                3_f32, 6_f32
            ],
            vec![
                1_f32, 3_f32, 5_f32,
                2_f32, 4_f32, 6_f32,
            ],
        ];

        for i in 0..input_list.len() {
            let (width, height, ref input) = input_list[i];

            println!("");
            let mut output = input.clone();
            transpose(width, height, input.as_slice(), output.as_mut_slice());


            println!("input:   {:?}", input);
            println!("actual:  {:?}", output);
            println!("expected:{:?}", expected_list[i]);

            compare_float_vectors(&expected_list[i].as_slice(), &output.as_slice());
        }
    }
}
