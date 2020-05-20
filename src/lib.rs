
// Parallel processing
use rayon::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
pub struct PixelData {
    pub x: i32,
    pub y: i32,
    pub v: Vec<f32>,
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
    
    (0..size * size)
        .map(|i| {
            let v = (0..dimension).map(|d| img[(i + d * (size * size)) as usize]).collect::<Vec<_>>();
            PixelData {
                x: i % size,
                y: i / size,
                v,
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
    dimension: i32,
    factor: f32,
    pixels: &Vec<PixelData>,
    p1: &PixelData,
    p2: &PixelData,
) -> Result {
    let dimension = dimension as usize;
    pixels
        .par_iter()
        .map(|p| {
            // Here compute the cost
            // --- Distance (with cycle mapping)
            let dist_1_x = (p.x - p1.x).abs();
            let dist_1_y = (p.y - p1.y).abs();
            let sqr_dist_1 = dist_1_x.min(size - dist_1_x).pow(2) + dist_1_y.min(size - dist_1_y).pow(2);
            let dist_2_x = (p.x - p2.x).abs();
            let dist_2_y = (p.y - p2.y).abs();
            let sqr_dist_2 = dist_2_x.min(size - dist_2_x).pow(2) + dist_2_y.min(size - dist_2_y).pow(2);
            if (sqr_dist_1 == 0) || (sqr_dist_2 == 0) {
                Result {
                    org: 0.0,
                    new: 0.0,
                }
            } else {
                let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                let dist_term_2 = (-(sqr_dist_2 as f32) / SIGMA_I).exp();
                // --- Value
                let value_term_1 = (0..dimension).map(|d| (p1.v[d] - p.v[d]).abs()).sum::<f32>();
                let value_term_1 = (-value_term_1.powf(factor) / SIGMA_S).exp();
                let value_term_2 = (0..dimension).map(|d| (p2.v[d] - p.v[d]).abs()).sum::<f32>();
                let value_term_2 = (-value_term_2.powf(factor) / SIGMA_S).exp();
                Result {
                    org: dist_term_1 * value_term_1 + dist_term_2 * value_term_2,
                    new: dist_term_1 * value_term_2 + dist_term_2 * value_term_1,
                }
            }
        })
        .sum::<Result>()
}
pub fn energy(size: i32, dimension: i32, factor: f32, pixels: &Vec<PixelData>) -> f32 {
    let dimension = dimension as usize;
    pixels
        .par_iter()
        .map(|p1| {
            pixels
                .iter()
                .map(|p| {
                    let sqr_dist_1 = (p.x - p1.x).pow(2).min((p.x + size - p1.x).pow(2))
                        + (p.y - p1.y).pow(2).min((p.y + size - p1.y).pow(2));
                    if sqr_dist_1 == 0 {
                        0.0
                    } else {
                        let dist_term_1 = (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                        let value_term_1 = (0..dimension).map(|d| (p1.v[d] - p.v[d]).abs()).sum::<f32>();
                        let value_term_1 = (-value_term_1.powf(factor) / SIGMA_S).exp();
                        dist_term_1 * value_term_1
                    }
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        * 0.5
}


pub fn cluser_and_void(size: i32, dimension: i32, p: (i32, i32, i32)) -> Vec<PixelData> {
    #[derive(Clone, Debug)]
    struct MapEntry {
        w: f32,
        rank: i32,
    }
    impl Default for MapEntry {
        fn default() -> Self {
            Self {
                w: 0.0,
                rank: -1,
            }
        }
    }
    let get_index =
        |p: (i32, i32, i32)| -> usize { (p.2 * size * size + p.1 * size + p.0) as usize };
    
    let splat = |p: (i32, i32, i32), map: &mut Vec<MapEntry> | -> (i32, i32, i32) {
        let mut min_v = std::f32::MAX;
        let mut min_i = (-1, -1, -1);
        for d in 0..dimension {
            for y in 0..size {
                for x in 0..size {
                    // Update weight
                    let dist_1_x = (p.0 - x).abs();
                    let dist_1_y = (p.1 - y).abs();
                    let dist_1_d = (p.2 - d).abs();
                    let sqr_dist_1 = dist_1_x.min(size - dist_1_x).pow(2) + dist_1_y.min(size - dist_1_y).pow(2) + dist_1_d.min(dimension - dist_1_d).pow(2);
                    let c = get_index((x,y,d));
                    map[c].w += (-(sqr_dist_1 as f32) / SIGMA_I).exp();
                    if map[c].w < min_v && map[c].rank == -1 {
                        min_i = (x,y,d);
                        min_v = map[c].w;
                    }
                }
            }
        }
        min_i
    };

    let mut current = p;
    let mut map = vec![MapEntry::default(); (size * size * dimension) as usize];
    for i in 0..(size * size * dimension) {
        map[get_index(current)].rank = i;
        current = splat(current, &mut map);
    }

    // Compute renormalize map
    let min_max = (0..dimension as usize).map(|d| {
        let size_2 = (size * size) as usize;
        let min = (size_2*d..size_2*(d+1)).map(|i| map[i].rank).min().unwrap();
        let max = (size_2*d..size_2*(d+1)).map(|i| map[i].rank).max().unwrap();
        (min, max)
    }).collect::<Vec<_>>();

    (0..size*size).map(|i| {
        let x = i % size;
        let y = i / size;
        PixelData {
            x,
            y,
            v: (0..dimension).map(|d| {
                ((map[get_index((x,y,d))].rank - min_max[d as usize].0) as f32 + 0.5) / (min_max[d as usize].1 - min_max[d as usize].0) as f32
            }).collect::<Vec<_>>()
        }
    }).collect::<Vec<_>>()
}

pub mod fft;