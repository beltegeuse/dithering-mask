
// Parallel processing
use rayon::prelude::*;
use rand::prelude::*;

pub struct PixelData {
    pub x: i32,
    pub y: i32,
    pub d: i32,
    pub v: f32,
}
pub fn gen_pixels(size: i32, dimension: i32, mut rng: &mut rand::rngs::StdRng) -> Vec<PixelData> {
    let mut img = vec![0.0 as f32; (size * size * dimension) as usize];
    let inv_2d_size = 1.0 / (size * size) as f32;
    for d in 0..dimension {
        let offset_rand = rng.gen_range(0.0, 1.0) * inv_2d_size;
        let offset_index = (d * (size * size)) as usize;
        for i in 0..(size * size) as usize {
            img[i + offset_index] = (i as f32 + offset_rand) / (size * size) as f32;
        }
    }
    img.shuffle(&mut rng);
    
    (0..size * size * dimension)
        .map(|v| {
            let v2 = v % (size * size);
            PixelData {
                x: v2 % size,
                y: v2 / size,
                d: v / (size * size),
                v: img[v as usize],
            }
        })
        .collect::<Vec<_>>()
}

// Lambda that compute the energy if the pixel get swap or not
// This information is used later for computing the probability to swap
pub struct Result {
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
const SIGMA_I: f32 = 2.1 * 2.1;
const SIGMA_S: f32 = 1.0;
pub fn energy_diff_pair(
    size: i32,
    factor: f32,
    pixels: &Vec<PixelData>,
    p1: &PixelData,
    p2: &PixelData,
) -> Result {
    pixels
        .par_iter()
        .map(|p| {
            // Here compute the cost
            // --- Distance (with cycle mapping)
            let sqr_dist_1 = (p.x - p1.x).pow(2).min((p.x + size - p1.x).pow(2))
                + (p.y - p1.y).pow(2).min((p.y + size - p1.y).pow(2));
            let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
            let sqr_dist_2 = (p.x - p2.x).pow(2).min((p.x + size - p2.x).pow(2))
                + (p.y - p2.y).pow(2).min((p.y + size - p2.y).pow(2));
            let dist_term_2 = (-(sqr_dist_2 as f32) / SIGMA_I).exp();
            // --- Value
            let value_term_1 = (-(p1.v - p.v).abs().powf(factor) / SIGMA_S).exp();
            let value_term_2 = (-(p2.v - p.v).abs().powf(factor) / SIGMA_S).exp();
            Result {
                org: dist_term_1 * value_term_1 + dist_term_2 * value_term_2,
                new: dist_term_1 * value_term_2 + dist_term_2 * value_term_1,
            }
        })
        .sum::<Result>()
}
pub fn energy(size: i32, factor: f32, pixels: &Vec<PixelData>) -> f32 {
    pixels
        .par_iter()
        .map(|p1| {
            pixels
                .iter()
                .map(|p| {
                    let sqr_dist_1 = (p.x - p1.x).pow(2).min((p.x + size - p1.x).pow(2))
                        + (p.y - p1.y).pow(2).min((p.y + size - p1.y).pow(2));
                    let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                    let value_term_1 = (-(p1.v - p.v).abs().powf(factor) / SIGMA_S).exp();
                    dist_term_1 * value_term_1
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        * 0.5
}

pub mod fft;