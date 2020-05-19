// FFT
pub fn fft1d(real: &[f32], img: &[f32], out_real: &mut [f32], out_img: &mut [f32], size: usize) {
    let inv_size = 1.0 / size as f32;

    // For the sum
    struct ResultFFT {
        pub real: f32,
        pub img: f32,
    }
    impl std::iter::Sum for ResultFFT {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            iter.fold(
                Self {
                    real: 0.0,
                    img: 0.0,
                },
                |a, b| Self {
                    real: a.real + b.real,
                    img: a.img + b.img,
                },
            )
        }
    }

    // Compute FFT
    for i in 0..size {
        let constant = 2.0 * std::f32::consts::PI * i as f32 * inv_size;
        let res = (0..size)
            .map(|j| {
                let cos_constant = (j as f32 * constant).cos();
                let sin_constant = (j as f32 * constant).sin();
                ResultFFT {
                    real: real[j] * cos_constant + img[j] * sin_constant,
                    img: -real[j] * sin_constant + img[j] * cos_constant,
                }
            })
            .sum::<ResultFFT>();
        out_real[i] = res.real * inv_size;
        out_img[i] = res.img * inv_size;
    }
}

pub fn fft2d(real: &[f32], size: usize) -> Vec<f32> {
    let size_sqr = size * size;

    let mut real_temp_1 = (0..size_sqr)
        .map(|i| real[i] * (-1.0 as f32).powi(((i % size) + (i / size)) as i32))
        .collect::<Vec<f32>>();
    let mut img_temp_1 = vec![0.0; size_sqr];
    let mut real_temp_2 = vec![0.0; size_sqr];
    let mut img_temp_2 = vec![0.0; size_sqr];

    // Horizontal
    for i in 0..size {
        let index = i * size;
        fft1d(
            &real_temp_1[index..index + size],
            &img_temp_1[index..index + size],
            &mut real_temp_2[index..index + size],
            &mut img_temp_2[index..index + size],
            size,
        );
    }

    // Rotate image & Vertical
    for i in 0..size_sqr {
        real_temp_1[i] = real_temp_2[(i % size) * size + (i / size)];
        img_temp_1[i] = img_temp_2[(i % size) * size + (i / size)];
    }
    for i in 0..size {
        let index = i * size;
        fft1d(
            &real_temp_1[index..index + size],
            &img_temp_1[index..index + size],
            &mut real_temp_2[index..index + size],
            &mut img_temp_2[index..index + size],
            size,
        );
    }

    // Normalize and output
    (0..size_sqr)
        .map(|i| {
            ((real_temp_2[i] * real_temp_2[i] + img_temp_2[i] * img_temp_2[i]).sqrt() + 1.0).ln()
                * size as f32
                * 2.0
        })
        .collect::<Vec<f32>>()
}