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
use dithering_mask;

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
            .get_matches();

    // Reading user parameters
    // -- Global
    let size = value_t_or_exit!(matches.value_of("size"), i32);
    let dimension = value_t_or_exit!(matches.value_of("dimension"), i32);
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

    // -- Debug (if necessary)
    let output_iter = match matches.value_of("intermediate") {
        None => None,
        Some(v) => Some(v.parse::<i32>().expect(&format!("{} is not an i32", v))),
    };
    let fft = matches.is_present("fft");

    // Helper to convert (x,y,d) to img coordinates
    let get_index =
    |p: (i32, i32)| -> usize { (p.1 * size + p.0) as usize };

    let nb_moves_per_iteration = size*size;
    let mut nb_iterations = 0; // Count the number of iteration (effective, discard auto-tunned iter)
    let mut temperature = 10.0; // Very high temperature (but auto-tuned)
    let cooling_rate = 0.001_f64;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    // Do anneling for higher dimension, otherwise, cluster and void
    let anneling = dimension != 1;
    let pixels = if anneling {
        //////////////////////
        // Simulated anneling
        //////////////////////

        // Create the thread pool
        let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        // Create the random image
        let mut pixels = dithering_mask::gen_pixels(size, dimension, &mut rng);

        // Compute the total energy
        // Note that this information is informative only
        // as total energy is never used in our optimization process
        info!("Compute total energy ...");
        let mut energy = dithering_mask::energy(size, dimension, factor, &pixels);
        info!("Total energy: {}", energy);

        while nb_iterations < iteration {
            // For computing how long take an iteration...
            let now = time::Instant::now();
            // Save a temporay image if needed
            if let Some(iter) = output_iter {
                let img = (0..dimension as usize).map(|d| pixels.iter().map(|p| p.v[d]).collect::<Vec<_>>()).collect::<Vec<_>>();
                if (nb_iterations+1) % iter == 0 {
                    if dimension == 2 {
                        save_img_2d(
                            &img[0],
                            &img[1],
                            (size as usize, size as usize),
                            &format!("{}_{}.{}", output, nb_iterations+1, ext),
                        );

                        if fft {
                            let fft_r = dithering_mask::fft::fft2d(&img[0], size as usize);
                            let fft_g = dithering_mask::fft::fft2d(&img[1], size as usize);
                            save_img_2d(
                                &fft_r[..],
                                &fft_g[..],
                                (size as usize, size as usize),
                                &format!("{}_fft_{}.{}", output, nb_iterations+1, ext),
                            );
                        }
                    } else {
                        // Take the first dimension by default
                        save_img(
                            &img[0],
                            (size as usize, size as usize),
                            &format!("{}_{}.{}", output, nb_iterations+1, ext),
                        );

                        if fft {
                            let fft_r = dithering_mask::fft::fft2d(&img[1], size as usize);
                            save_img(
                                &fft_r[..],
                                (size as usize, size as usize),
                                &format!("{}_fft_{}.{}", output, nb_iterations+1, ext),
                            );
                        }
                    }
                }
            }

            // Make the cooling factor independent to the image size
            let total_swaps = nb_iterations * nb_moves_per_iteration;
            let total_swaps_norm = total_swaps as f64 / ((size * size) as f64).sqrt();
            let cooling_factor = (1.0_f64 - cooling_rate).powf(total_swaps_norm);

            // Shuffle the pair
            // At each iteration, all pixel will have a swap attempt
            info!(
                "[{}/{}] \t Energy: {} \t Temp: {}",
                nb_iterations+1,
                iteration,
                energy,
                cooling_factor * temperature
            );

            // Do the swapping in 2D
            let mut nb_accepted_move = 0;
            for _ in 0..(size*size) {
                // Generate two random different 2D coordinate 
                let p_1 = (rng.gen_range(0..size), rng.gen_range(0..size));
                let p_2 = {
                    let mut p_2 = (rng.gen_range(0..size), rng.gen_range(0..size));
                    while p_1.0 == p_2.0 && p_1.1 == p_2.1 {
                        p_2 = (rng.gen_range(0..size), rng.gen_range(0..size));
                    }
                    p_2
                };
                
                // Compute the new and old energy if we was doing the swap
                let res = pool.install(|| {
                    dithering_mask::energy_diff_pair(
                        size,
                        dimension,
                        factor,
                        &pixels,
                        &pixels[get_index(p_1)],
                        &pixels[get_index(p_2)],
                    )
                });
                
                let delta = res.new - res.org;
                let accept_prob = (- delta as f64 / (cooling_factor * temperature)).exp();
                // If we want to do the swap
                if accept_prob > rng.gen_range(0.0..1.0) {
                    // Swap img values
                    let tmp = pixels[get_index(p_1)].v.clone();
                    pixels[get_index(p_1)].v = pixels[get_index(p_2)].v.clone();
                    pixels[get_index(p_2)].v = tmp;
                    // Update the state
                    energy += delta;
                    nb_accepted_move += 1;
                }
                
            }

            let acceptance_rate = nb_accepted_move as f32 / nb_moves_per_iteration as f32;
            if nb_iterations == 0 && acceptance_rate > 0.40 {
                info!("Decrease the inital temperature... Retry the first iteration!");
                temperature *= 0.5;
            } else if acceptance_rate > 0.0 {
                nb_iterations += 1;
            }

            // Update temperature for the next iteration
            info!(
                "Accept rate: {} \t Time: {} sec",
                acceptance_rate * 100.0,
                now.elapsed().as_secs_f32()
            );
        }
        pixels
    } else {
        assert!(dimension == 1);
        dithering_mask::cluser_and_void(size, (rng.gen_range(0..size),rng.gen_range(0..size)))
    };

    // Save final (all dimension)
    let img = (0..dimension as usize).map(|d| pixels.iter().map(|p| p.v[d]).collect::<Vec<_>>()).collect::<Vec<_>>();
    match dimension {
        1 => {
            save_img(
                &img[0],
                (size as usize, size as usize),
                &format!("{}.{}", output, ext),
            );
            if fft {
                let fft_r = dithering_mask::fft::fft2d(&img[0], size as usize);
                save_img(
                    &fft_r,
                    (size as usize, size as usize),
                    &format!("{}_fft.{}", output, ext),
                );
            }
        }
        2 => {
            save_img_2d(
                &img[0],
                &img[1],
                (size as usize, size as usize),
                &format!("{}.{}", output, ext),
            );
            if fft {
                let fft_r = dithering_mask::fft::fft2d(&img[0], size as usize);
                let fft_g = dithering_mask::fft::fft2d(&img[1], size as usize);
                save_img_2d(
                    &fft_r,
                    &fft_g,
                    (size as usize, size as usize),
                    &format!("{}_fft.{}", output, ext),
                );
            }
        }
        _ => {
            for d in 0..dimension as usize {
                save_img(
                    &img[d],
                    (size as usize, size as usize),
                    &format!("{}_dim_{}.{}", output, d, ext),
                );

                if fft {
                    let fft_r = dithering_mask::fft::fft2d(&img[d], size as usize);
                    save_img(
                        &fft_r,
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
                    let p = img[d as usize][get_index((x, y))];
                    file.write_f32::<LittleEndian>(p.abs()).unwrap();
                }
            }
        }
    }
}
