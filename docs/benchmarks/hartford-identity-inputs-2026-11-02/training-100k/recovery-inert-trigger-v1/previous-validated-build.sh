#!/bin/bash
set -euo pipefail
source /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/env.sh
export PATH=/home/altf4/benchmarks/hartford-compact-ppo-20261101/native-toolchain-loop-v1/bin:/home/altf4/benchmarks/hartford-compact-ppo-20261101/rust-toolchain-1.98.0/bin:$PATH
cd /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2
export PYTHONPATH=src:tests:.
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
export PATCHWORK_OBSERVATION_PIPELINE=python
export PATCHWORK_ROLLOUT_STORAGE=pickle
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=tensorflow
export PYTHONDONTWRITEBYTECODE=1
export RUSTMAGE_ROOT=/home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687
export MPLCONFIGDIR=/home/altf4/benchmarks/hartford-compact-ppo-20261101/mpl-cache
export CARGO_TARGET_DIR=/home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/target-4bedacf19687-capped256
export CARGO_HOME=/home/altf4/benchmarks/hartford-compact-ppo-20261101/cargo-home-loop-v1
/usr/bin/python3 /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/scripts/resource_guard.py -- cargo generate-lockfile --manifest-path /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/native/direct-standard/Cargo.toml
/usr/bin/python3 /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/scripts/resource_guard.py -- cargo test --locked --manifest-path /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/Cargo.toml --test mandatory_loops
/usr/bin/python3 /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/scripts/resource_guard.py -- cargo build --release --locked --manifest-path /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/Cargo.toml --lib
/usr/bin/python3 /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/rustmage-4bedacf19687/scripts/resource_guard.py -- cargo fetch --locked --manifest-path /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/native/direct-standard/Cargo.toml
/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python /home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/scripts/build-direct-standard.py
