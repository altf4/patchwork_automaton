#!/usr/bin/env bash
set -euo pipefail
D=/home/altf4/benchmarks/direct-profile-20261031
S=/home/altf4/benchmarks/standard-direct-default-20261031/source
V=/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python
cd "$S"
export PYTHONPATH="$D:src:." DIRECT_STANDARD_LIBRARY="$S/.local/direct-standard/target/release/libpatchwork_direct_standard.so"
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 PATCHWORK_ROLLOUT_STORAGE=pickle PATCHWORK_STANDARD_PLAYBACK=direct
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 KERAS_BACKEND=tensorflow
exec "$V" -u /home/altf4/benchmarks/standard-ppo-100000-compact-20261028/memory-fix/guarded-launch.py \
 --root "$D/$1-guard" --command "$D/$1-command.json" --min-available-mib 1536 --memory-grace-seconds 5 --check-seconds 2
