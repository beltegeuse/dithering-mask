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

// Functions to write output mask
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

fn main() {
    // Setup logger
    env_logger::Builder::from_default_env()
        .format_timestamp(None)
        .parse_filters("info")
        .init();

    // Constants
    // TODO: Use clap for make it more user friendly
    const SIZE: i32 = 32;
    const ITER: i32 = 1024;
    const SAVE_ITER: i32 = 32;
    const DIM: i32 = 2;
    const SIGMA_I: f32 = 2.1;
    const SIGMA_S: f32 = 1.0;
    const FACTOR: f32 = DIM as f32 / 2.0;
    let slice_size = (SIZE * SIZE) as usize;
    let mut rng = rand::thread_rng();

    // Create the random image
    let mut img = vec![0.0 as f32; (SIZE * SIZE * DIM) as usize];
    for i in 0..(SIZE * SIZE * DIM) as usize {
        img[i] = rng.gen_range(0.0, 1.0);
    }
    info!("{:?}", img);

    // Create vector of pixel coordinate [(x,y,d), ...]
    let mut index = (0..SIZE * SIZE * DIM)
        .map(|v| {
            let v2 = v % (SIZE * SIZE);
            (v2 % SIZE, v2 / SIZE, v / (SIZE * SIZE))
        })
        .collect::<Vec<_>>();

    // Helper to convert (x,y,d) to img coordinates
    let get_index =
        |p: (i32, i32, i32)| -> usize { (p.2 * SIZE * SIZE + p.1 * SIZE + p.0) as usize };

    // Lambda that compute the energy if the pixel get swap or not
    // This information is used later for computing the probability to swap
    let energy_diff_pair =
        |img: &Vec<f32>, p1: (i32, i32, i32), p2: (i32, i32, i32)| -> (f32, f32) {
            let mut e_org = 0.0;
            let mut e_new = 0.0;
            let p1_v = img[get_index(p1)];
            let p2_v = img[get_index(p2)];
            for d in 0..DIM {
                for y in 0..SIZE {
                    for x in 0..SIZE {
                        if x == p1.0 && y == p1.1 && d == p1.2 {
                            continue;
                        }
                        if x == p2.0 && y == p2.1 && d == p2.2 {
                            continue;
                        }

                        let current_v = img[get_index((x, y, d))];

                        // Here compute the cost
                        // --- Distance
                        let sqr_dist_1 = (x - p1.0).pow(2) + (y - p1.1).pow(2) + (d - p1.2).pow(2);
                        let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                        let sqr_dist_2 = (x - p2.0).pow(2) + (y - p2.1).pow(2) + (d - p2.2).pow(2);
                        let dist_term_2 = (-(sqr_dist_2 as f32) / SIGMA_I).exp();
                        // --- Value
                        let value_term_1 = (-(p1_v - current_v).abs().powf(FACTOR) / SIGMA_S).exp();
                        let value_term_2 = (-(p2_v - current_v).abs().powf(FACTOR) / SIGMA_S).exp();

                        // Compute the orginal and the new value
                        e_org += dist_term_1 * value_term_1;
                        e_org += dist_term_2 * value_term_2;
                        e_new += dist_term_1 * value_term_2;
                        e_new += dist_term_2 * value_term_1;
                    }
                }
            }
            (e_new, e_org)
        };

    // Compute the energy for a given pixel
    // to all the other pixel of the image 
    let energy_pixel = |p1: (i32, i32, i32)| -> f32 {
        let p1_v = img[get_index(p1)];
        let mut e = 0.0;
        for d in 0..DIM {
            for y in 0..SIZE {
                for x in 0..SIZE {
                    if x == p1.0 && y == p1.1 && d == p1.2 {
                        continue;
                    }
                    let current_v = img[get_index((x, y, d))];

                    // Here compute the cost
                    // --- Distance
                    let sqr_dist_1 = (x - p1.0).pow(2) + (y - p1.1).pow(2) + (d - p1.2).pow(2);
                    let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                    let value_term_1 = (-(p1_v - current_v).abs().powf(FACTOR) / SIGMA_S).exp();
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
    let mut energy = 0.0;
    for d in 0..DIM {
        for y in 0..SIZE {
            for x in 0..SIZE {
                energy += energy_pixel((x, y, d));
            }
        }
    }
    energy /= 2.0; // TODO: For now pair are counted twice
    info!("Total energy: {}", energy);

    //////////////////////
    // Simulated anneling
    let mut delta_avg: f32 = 1.0;
    let mut total_accepted = 0;
    // Temperature 
    const P_1: f32 = 0.7; // Initial prob to accept a swap
    const P_FINAL: f32 = 0.001; // Final prob to accept a swap 
    let t_1: f32 = -1.0 / P_1.ln();
    let t50: f32 = -1.0 / P_FINAL.ln();
    let frac: f32 = (t50 / t_1).powf(1.0 / (ITER - 1) as f32);
    let mut temp = t_1; // The current temperature
    for t in 0..ITER {
        // For computing how long take an iteration...
        let now = time::Instant::now();
        // Save a temporay image if needed   
        if t % SAVE_ITER == 0 {
            if DIM == 2 {
                save_pfm_2d(
                    &img[0..slice_size],
                    &img[slice_size..(2 * slice_size)],
                    (SIZE as usize, SIZE as usize),
                    &format!("{}_.pfm", t),
                );
            } else {
                // Take the first dimension by default
                save_pfm(
                    &img[0..(SIZE * SIZE) as usize],
                    (SIZE as usize, SIZE as usize),
                    &format!("{}_.pfm", t),
                );
            }
        }
        // Shuffle the pair 
        // At each iteration, all pixel will have a swap attempt
        index.shuffle(&mut rng);
        info!(
            "[{}/{}] \t Energy: {} \t Temp: {}",
            t + 1,
            ITER,
            energy,
            temp
        );
        let mut accepted_moves = 0;
        for chunk in index.chunks_exact(2) {
            // Compute the new and old energy if we was doing the swap
            let (new, old) = energy_diff_pair(&img, chunk[0], chunk[1]);
            let delta = new - old;
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
                let tmp = img[get_index(chunk[0])];
                img[get_index(chunk[0])] = img[get_index(chunk[1])];
                img[get_index(chunk[1])] = tmp;
                // Update the state
                energy += delta;
                accepted_moves += 1;
                delta_avg =
                    (delta_avg * total_accepted as f32 + delta_abs) / (total_accepted + 1) as f32;
                total_accepted += 1;
            }
        }

        // Update temperature for the next iteration
        temp *= frac;
        info!(
            "Accepted: {} \t Delta Avg: {} \t Time: {} sec",
            accepted_moves,
            delta_avg,
            now.elapsed().as_secs_f32()
        );
    }

    // Save final (all dimension)
    match DIM {
        1 => save_pfm(
            &img[0..slice_size],
            (SIZE as usize, SIZE as usize),
            &"final.pfm",
        ),
        2 => save_pfm_2d(
            &img[0..slice_size],
            &img[slice_size..2*slice_size],
            (SIZE as usize, SIZE as usize),
            &"final.pfm",
        ),
        _ => {
            for d in 0..DIM {
                let d_beg = (d * SIZE * SIZE) as usize;
                save_pfm(
                    &img[d_beg..d_beg + (SIZE * SIZE) as usize],
                    (SIZE as usize, SIZE as usize),
                    &format!("final_{}_.pfm", d),
                );
            }
        }
    }
}
