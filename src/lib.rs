
// Parallel processing
use rayon::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
pub struct PixelData {
    pub x: i32,
    pub y: i32,
    pub v: Vec<f32>,
}
  
/// Generate all initial pixel data.
/// For 1D, it is fully stratified. For higher dimension, it will not provide any improvement.
/// @TODO: Check the validity for higher dimension
pub fn gen_pixels(size: i32, dimension: i32, mut rng: &mut rand::rngs::StdRng) -> Vec<PixelData> {
    // Image are represented with 1D vec
    let mut img = std::vec::Vec::with_capacity((size * size * dimension) as usize);
    let inv_2d_size = 1.0 / (size * size) as f32;
    
    // Do all dimension independently
    for _ in 0..dimension {
        let mut img_dim = vec![0.0 as f32; (size * size) as usize];
        let offset_rand = rng.gen_range(0.0..1.0) * inv_2d_size;
        for i in 0..(size * size) as usize {
            img_dim[i] = (i as f32 + offset_rand) / (size * size) as f32;
        }
        img_dim.shuffle(&mut rng);
        img.append(& mut img_dim);
    }
    
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

/// This structure is used to return the energy level when attempting a swap
pub struct Result {
    pub org: f32, //< Original energy level
    pub new: f32, //< Energy level after the swap
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

/// Compute the energy difference when swapping p1 and p2
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

/// Compute the total energy
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

/// Cluster and void to generate 1D mask
pub fn cluser_and_void(size: i32, p: (i32, i32)) -> Vec<PixelData> {
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
        |p: (i32, i32)| -> usize { (p.1 * size + p.0) as usize };
    
    let splat = |p: (i32, i32), map: &mut Vec<MapEntry> | -> (i32, i32) {
        let res = map.par_iter_mut().enumerate().map(
            |(i, v)| {
                let x = i as i32 % size;
                let y = i as i32 / size;
                // Update weight
                let dist_1_x = (p.0 - x).abs();
                let dist_1_y = (p.1 - y).abs();
                let sqr_dist_1 = dist_1_x.min(size - dist_1_x).pow(2) + dist_1_y.min(size - dist_1_y).pow(2);
                v.w += (-(sqr_dist_1 as f32) / SIGMA_I).exp();

                if v.rank == -1 {
                    Some((v.w, (x,y)))
                } else {
                    None
                }
            }
        ).filter(Option::is_some).map(Option::unwrap).min_by(|a,b| a.0.partial_cmp(&b.0).unwrap());        
        res.unwrap().1
    };

    let mut pb = pbr::ProgressBar::new(((size * size) / 128) as u64);
    let mut current = p;
    let mut map = vec![MapEntry::default(); (size * size) as usize];
    for i in 0..(size * size) {
        map[get_index(current)].rank = i;
        if i != (size * size) - 1 {
            current = splat(current, &mut map);
        }
        if i % 128 == 0 {
            pb.inc();
        }
    }
    pb.finish_print("done");

    // Convert back
    (0..size*size).map(|i| {
        let x = i % size;
        let y = i / size;
        PixelData {
            x,
            y,
            v: vec![(map[get_index((x,y))].rank as f32 + 0.5) / (size * size) as f32],
        }
    }).collect::<Vec<_>>()
}

pub mod fft;