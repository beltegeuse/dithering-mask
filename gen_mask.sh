cargo run --release -- -f -p 0.4:0.001 -f 32 1 mask_32_1
cargo run --release -- -f -p 0.4:0.001 -f 32 2 mask_32_2
cargo run --release -- -f -p 0.4:0.001 -f 32 4 mask_32_4
cargo run --release -- -f -p 0.4:0.0001 -f 32 8 mask_32_8

cargo run --release -- -f -p 0.4:0.001 -f 64 1 mask_64_1
cargo run --release -- -f -p 0.4:0.001 -f 64 2 mask_64_2
cargo run --release -- -f -p 0.4:0.001 -f 64 4 mask_64_4
cargo run --release -- -f -p 0.4:0.0001 -f 64 8 mask_64_8

cargo run --release -- -f -p 0.4:0.001 -f 128 1 mask_128_1
cargo run --release -- -f -p 0.4:0.001 -f 128 2 mask_128_2
cargo run --release -- -f -p 0.4:0.001 -f 128 4 mask_128_4
cargo run --release -- -f -p 0.4:0.0001 -f 128 8 mask_128_8