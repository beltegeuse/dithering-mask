use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use dithering_mask;

fn criterion_benchmark(c: &mut Criterion) {
    const DIM: i32 = 2;
    const SIZE: i32 = 32;

    // Create the image
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let factor = DIM as f32 / 2.0;
    let pixels = dithering_mask::gen_pixels(SIZE, DIM, &mut rng);

    let mut group = c.benchmark_group("Energy");
    group.sample_size(10);
    group.bench_function("energy naive", |b| b.iter(|| dithering_mask::energy(SIZE, DIM, factor, &pixels)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);