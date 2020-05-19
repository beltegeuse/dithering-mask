// Logging system
extern crate env_logger;
#[macro_use]
extern crate log;
// random generator
use rand::prelude::*;
// writing PFM
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time;
// Command line
use clap::*;
// LDR image (png)
use image::*;
// Parallel processing
use rayon::prelude::*;

// FFT
pub fn fft1d(real: &[f32], img: &[f32], out_real: &mut [f32], out_img: &mut [f32], size: usize) {
    let inv_size = 1.0 / size as f32;

    // For the sum
    struct Result {
        pub real: f32,
        pub img: f32,
    }
    impl std::iter::Sum for Result {
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
                Result {
                    real: real[j] * cos_constant + img[j] * sin_constant,
                    img: -real[j] * sin_constant + img[j] * cos_constant,
                }
            })
            .sum::<Result>();
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

// Functions to write output mask
pub fn save_ldr_image(img: &[f32], size: (usize, usize), imgout_path_str: &str) {
    let mut image_ldr = DynamicImage::new_rgb8(size.0 as u32, size.1 as u32);
    for x in 0..size.0 {
        for y in 0..size.1 {
            let p = img[y * size.0 + x];
            image_ldr.put_pixel(
                x as u32,
                y as u32,
                Rgba::from_channels(
                    (p.min(1.0) * 255.0) as u8,
                    (p.min(1.0) * 255.0) as u8,
                    (p.min(1.0) * 255.0) as u8,
                    255,
                ),
            );
        }
    }
    image_ldr
        .save(&Path::new(imgout_path_str))
        .expect("failed to write img into file");
}
fn save_pfm(img: &[f32], size: (usize, usize), imgout_path_str: &str) {
    let file = File::create(Path::new(imgout_path_str)).unwrap();
    let mut file = BufWriter::new(file);
    let header = format!("PF\n{} {}\n-1.0\n", size.0, size.1);
    file.write_all(header.as_bytes()).unwrap();
    for y in 0..size.1 {
        for x in 0..size.0 {
            let p = img[y * size.0 + x];
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
        }
    }
}
fn save_pfm_2d(r: &[f32], g: &[f32], size: (usize, usize), imgout_path_str: &str) {
    let file = File::create(Path::new(imgout_path_str)).unwrap();
    let mut file = BufWriter::new(file);
    let header = format!("PF\n{} {}\n-1.0\n", size.0, size.1);
    file.write_all(header.as_bytes()).unwrap();
    for y in 0..size.1 {
        for x in 0..size.0 {
            let p_r = r[y * size.0 + x];
            let p_g = g[y * size.0 + x];
            file.write_f32::<LittleEndian>(p_r.abs()).unwrap();
            file.write_f32::<LittleEndian>(p_g.abs()).unwrap();
            file.write_f32::<LittleEndian>(0.0).unwrap();
        }
    }
}
pub fn save_ldr_image_2d(r: &[f32], g: &[f32], size: (usize, usize), imgout_path_str: &str) {
    let mut image_ldr = DynamicImage::new_rgb8(size.0 as u32, size.1 as u32);
    for x in 0..size.0 {
        for y in 0..size.1 {
            let r_v = r[y * size.0 + x];
            let g_v = g[y * size.0 + x];
            image_ldr.put_pixel(
                x as u32,
                y as u32,
                Rgba::from_channels(
                    (r_v.min(1.0) * 255.0) as u8,
                    (g_v.min(1.0) * 255.0) as u8,
                    0,
                    255,
                ),
            );
        }
    }
    image_ldr
        .save(&Path::new(imgout_path_str))
        .expect("failed to write img into file");
}

fn main() {
    // Setup logger
    env_logger::Builder::from_default_env()
        .format_timestamp(None)
        .parse_filters("info")
        .init();

    let matches =
        App::new("dithering-mask")
            .version("0.1.0")
            .author("Adrien Gruson <adrien.gruson@gmail.com>")
            .about("A Rusty Implementation of Mask generation - \"Blue-noise dithered sampling\", Georgiev and Fajardo, 2016")
            .arg(
                Arg::with_name("size")
                    .required(true)
                    .takes_value(true)
                    .help("image size"),
            )
            .arg(
                Arg::with_name("dimension")
                    .required(true)
                    .takes_value(true)
                    .help("mask dimension"),
            )
            .arg(
                Arg::with_name("output")
                    .required(true)
                    .takes_value(true)
                    .help("output name (.pfm)"),
            )
            .arg(
                Arg::with_name("intermediate")
                    .short("x")
                    .takes_value(true)
                    .help("frequency of intermediate output (debug)"),
            )
            .arg(
                Arg::with_name("iteration")
                    .short("i")
                    .takes_value(true)
                    .help("number of iterations")
                    .default_value("1024"),
            )
            .arg(
                Arg::with_name("png")
                    .short("l")
                    .help("save in png (otherwise save in pfm by default)"),
            )
            .arg(
                Arg::with_name("fft")
                    .short("f")
                    .help("do the fft"),
            )
            .arg(
                Arg::with_name("probabilities")
                    .short("p")
                    .takes_value(true)
                    .help("probability to accept worst move (init:final)")
                    .default_value("0.1:0.001"),
            )
            .get_matches();

    // Reading user parameters
    // -- Global
    let size = value_t_or_exit!(matches.value_of("size"), i32);
    let dimension = value_t_or_exit!(matches.value_of("dimension"), i32);
    let slice_size = (size * size) as usize;
    let factor = dimension as f32 / 2.0;
    // --- image output
    let output = value_t_or_exit!(matches.value_of("output"), String);
    let ext = if matches.is_present("png") {
        "png"
    } else {
        "pfm"
    };
    let save_img = if matches.is_present("png") {
        save_ldr_image
    } else {
        save_pfm
    };
    let save_img_2d = if matches.is_present("png") {
        save_ldr_image_2d
    } else {
        save_pfm_2d
    };
    // -- Optimization
    let iteration = value_t_or_exit!(matches.value_of("iteration"), i32);
    let probability = value_t_or_exit!(matches.value_of("probabilities"), String);
    let probability = probability
        .split(":")
        .into_iter()
        .map(|v| v.parse::<f32>().expect("f32 for probabilities"))
        .collect::<Vec<_>>();
    if probability.len() != 2 {
        panic!("the temperature have to be formated as: 0.01:0.001 (init:end)");
    }
    if probability[0] < probability[1] {
        panic!(
            "First probability {} need to be higher to final one {}",
            probability[0], probability[1]
        );
    }
    // -- Debug
    let output_iter = match matches.value_of("intermediate") {
        None => None,
        Some(v) => Some(v.parse::<i32>().expect(&format!("{} is not an i32", v))),
    };
    let fft = matches.is_present("fft");
    // -- Other constants
    const SIGMA_I: f32 = 2.1 * 2.1;
    const SIGMA_S: f32 = 1.0;

    // Create the thread pool
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    // Create the random image
    let mut rng = rand::thread_rng();
    let mut img = vec![0.0 as f32; (size * size * dimension) as usize];
    {
        let inv_2d_size = 1.0 / (size * size) as f32;
        for d in 0..dimension {
            let offset_rand = rng.gen_range(0.0, 1.0) * inv_2d_size;
            let offset_index = (d * (size * size)) as usize;
            for i in 0..(size * size) as usize {
                img[i+offset_index] = (i as f32 + offset_rand) / (size * size) as f32;
            }
        }
        img.shuffle(&mut rng);
    }
    // Create vector of pixel coordinate [(x,y,d), ...]
    let index_order = (0..size * size * dimension)
    .map(|v| {
        let v2 = v % (size * size);
        (v2 % size, v2 / size, v / (size * size))
    })
    .collect::<Vec<_>>();
    // Only 2D index, as we need to make a special treatment
    // for dimension
    let mut index_2d = (0..size * size)
    .map(|v| {
        let v2 = v % (size * size);
        (v2 % size, v2 / size)
    })
    .collect::<Vec<_>>();

    // Helper to convert (x,y,d) to img coordinates
    let get_index =
        |p: (i32, i32, i32)| -> usize { (p.2 * size * size + p.1 * size + p.0) as usize };

    // Lambda that compute the energy if the pixel get swap or not
    // This information is used later for computing the probability to swap
    struct Result {
        pub org: f32,
        pub new: f32,
    }
    impl std::iter::Sum for Result {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            iter.fold(Self { org: 0.0, new: 0.0 }, |a, b| Self {
                org: a.org + b.org,
                new: a.new + b.new,
            })
        }
    }
    let energy_diff_pair = |img: &Vec<f32>, p1: (i32, i32, i32), p2: (i32, i32, i32)| -> Result {
        let p1_v = img[get_index(p1)];
        let p2_v = img[get_index(p2)];
        pool.install(|| {
            index_order
                .par_iter()
                .map(|(x, y, d)| {
                    // TODO: Avoid this by zipping values inside
                    // the index order.
                    let current_v = img[get_index((*x, *y, *d))];
                    // Here compute the cost
                    // --- Distance (with cycle mapping)
                    let sqr_dist_1 = (x - p1.0).abs().min((x + size - p1.0).abs()).pow(2)
                        + (y - p1.1).abs().min((y + size - p1.1).abs()).pow(2);
                    let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                    let sqr_dist_2 = (x - p2.0).abs().min((x + size - p2.0).abs()).pow(2)
                        + (y - p2.1).abs().min((y + size - p2.1).abs()).pow(2);
                    let dist_term_2 = (-(sqr_dist_2 as f32) / SIGMA_I).exp();
                    // --- Value
                    let value_term_1 = (-(p1_v - current_v).abs().powf(factor) / SIGMA_S).exp();
                    let value_term_2 = (-(p2_v - current_v).abs().powf(factor) / SIGMA_S).exp();

                    Result {
                        org: dist_term_1 * value_term_1 + dist_term_2 * value_term_2,
                        new: dist_term_1 * value_term_2 + dist_term_2 * value_term_1,
                    }
                })
                .sum::<Result>()
        })
    };

    // Compute the energy for a given pixel
    // to all the other pixel of the image
    let energy_pixel = |p1: (i32, i32, i32)| -> f32 {
        let p1_v = img[get_index(p1)];
        let mut e = 0.0;
        for d in 0..dimension {
            for y in 0..size {
                for x in 0..size {
                    let current_v = img[get_index((x, y, d))];
                    // Here compute the cost
                    // --- Distance
                    let sqr_dist_1 = (x - p1.0).abs().min((x + size - p1.0).abs()).pow(2)
                        + (y - p1.1).abs().min((y + size - p1.1).abs()).pow(2);
                    let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                    let value_term_1 = (-(p1_v - current_v).abs().powf(factor) / SIGMA_S).exp();
                    // Compute the orginal and the new value
                    e += dist_term_1 * value_term_1;
                }
            }
        }
        e
    };

    // Compute the total energy
    // Note that this information is informative only
    // as total energy is never used in our optimization process
    info!("Compute total energy ...");
    let mut energy = index_order
        .par_iter()
        .map(|(x, y, d)| energy_pixel((*x, *y, *d)))
        .sum::<f32>()
        * 0.5;
    info!("Total energy: {}", energy);

    //////////////////////
    // Simulated anneling
    let mut delta_avg: f32 = 1.0;
    let mut total_accepted = 0;
    // Temperature
    let t_1: f32 = -1.0 / probability[0].ln();
    let t50: f32 = -1.0 / probability[1].ln();
    let frac: f32 = (t50 / t_1).powf(1.0 / (iteration - 1) as f32);
    let mut temp = t_1; // The current temperature
    for t in 0..iteration {
        // For computing how long take an iteration...
        let now = time::Instant::now();
        // Save a temporay image if needed
        if let Some(iter) = output_iter {
            if t % iter == 0 {
                if dimension == 2 {
                    save_img_2d(
                        &img[0..slice_size],
                        &img[slice_size..(2 * slice_size)],
                        (size as usize, size as usize),
                        &format!("{}_{}.{}", output, t, ext),
                    );

                    if fft {
                        let fft_r = fft2d(&img[0..slice_size], size as usize);
                        let fft_g = fft2d(&img[slice_size..2 * slice_size], size as usize);
                        save_img_2d(
                            &fft_r[..],
                            &fft_g[..],
                            (size as usize, size as usize),
                            &format!("{}_fft_{}.{}", output, t, ext),
                        );
                    }
                } else {
                    // Take the first dimension by default
                    save_img(
                        &img[0..(size * size) as usize],
                        (size as usize, size as usize),
                        &format!("{}_{}.{}", output, t, ext),
                    );

                    if fft {
                        let fft_r = fft2d(&img[0..slice_size], size as usize);
                        save_img(
                            &fft_r[..],
                            (size as usize, size as usize),
                            &format!("{}_fft_{}.{}", output, t, ext),
                        );
                    }
                }
            }
        }
        // Shuffle the pair
        // At each iteration, all pixel will have a swap attempt
        index_2d.shuffle(&mut rng);
        info!(
            "[{}/{}] \t Energy: {} \t Temp: {}",
            t + 1,
            iteration,
            energy,
            temp
        );
        let mut accepted_moves = 0;
        // Important: We only swap pair on the same dimension
        // indeed, otherwise, the distribution can change in different dimension
        // leading to problem when optimizing large dimension mask. 
        // Note that the optimization procedure will be less effective when
        // the dimension of the mask increases
        for d in 0..dimension {
            for chunk in index_2d.chunks_exact(2) {
                let p_1 = (chunk[0].0, chunk[0].1, d);
                let p_2 = (chunk[1].0, chunk[1].1, d);
                
                // Compute the new and old energy if we was doing the swap
                let res = energy_diff_pair(&img, p_1, p_2);
                let delta = res.new - res.org;
                let delta_abs = delta.abs();
                let accept = if delta < 0.0 {
                    true
                } else {
                    // Compute acceptence probability with temperature
                    let a = (-delta_abs / (delta_avg * temp)).exp();
                    a >= rng.gen_range(0.0, 1.0)
                };

                // If we want to do the swap
                if accept {
                    // Swap img values
                    let tmp = img[get_index(p_1)];
                    img[get_index(p_1)] = img[get_index(p_2)];
                    img[get_index(p_2)] = tmp;
                    // Update the state
                    energy += delta;
                    accepted_moves += 1;
                    delta_avg =
                        (delta_avg * total_accepted as f32 + delta_abs) / (total_accepted + 1) as f32;
                    total_accepted += 1;
                }
            }
        }

        // Update temperature for the next iteration
        temp *= frac;
        info!(
            "Accept rate: {} \t Delta Avg: {} \t Time: {} sec",
            (accepted_moves as f32 / (index_order.len()) as f32 * 50.0),
            delta_avg,
            now.elapsed().as_secs_f32()
        );
    }

    // Save final (all dimension)
    match dimension {
        1 => {
            save_img(
                &img[0..slice_size],
                (size as usize, size as usize),
                &format!("{}.{}", output, ext),
            );
            if fft {
                let fft_r = fft2d(&img[0..slice_size], size as usize);
                save_img(
                    &fft_r[..],
                    (size as usize, size as usize),
                    &format!("{}_fft.{}", output, ext),
                );
            }
        }
        2 => {
            save_img_2d(
                &img[0..slice_size],
                &img[slice_size..2 * slice_size],
                (size as usize, size as usize),
                &format!("{}.{}", output, ext),
            );
            if fft {
                let fft_r = fft2d(&img[0..slice_size], size as usize);
                let fft_g = fft2d(&img[slice_size..2 * slice_size], size as usize);
                save_img_2d(
                    &fft_r[..],
                    &fft_g[..],
                    (size as usize, size as usize),
                    &format!("{}_fft.{}", output, ext),
                );
            }
        }
        _ => {
            for d in 0..dimension {
                let d_beg = (d * size * size) as usize;
                save_img(
                    &img[d_beg..d_beg + (size * size) as usize],
                    (size as usize, size as usize),
                    &format!("{}_dim_{}.{}", output, d, ext),
                );

                if fft {
                    let fft_r = fft2d(&img[d_beg..d_beg + (size * size) as usize], size as usize);
                    save_img(
                        &fft_r[..],
                        (size as usize, size as usize),
                        &format!("{}_fft_dim_{}.{}", output, d, ext),
                    );
                }
            }
        }
    }

    // Dump output (float map)
    // This is a custom format to easily load in other programs.
    {
        let file = File::create(Path::new(&format!("{}.mask", output))).unwrap();
        let mut file = BufWriter::new(file);
        let header = format!("MASK\n{} {} {}\n-1.0\n", size, size, dimension);
        file.write_all(header.as_bytes()).unwrap();
        for d in 0..dimension {
            for y in 0..size {
                for x in 0..size {
                    let p = img[get_index((x, y, d))];
                    file.write_f32::<LittleEndian>(p.abs()).unwrap();
                }
            }
        }
    }
}
