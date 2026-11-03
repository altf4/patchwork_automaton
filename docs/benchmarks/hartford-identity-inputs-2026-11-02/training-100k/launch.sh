#!/bin/bash
set -euo pipefail
source /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/env.sh
export PYTHONPATH=/home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1/src:/home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1/tests:/home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1
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
export RUSTMAGE_LIBRARY=/home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/target-4bedacf19687-capped256/release/librustmage.so
export DIRECT_STANDARD_LIBRARY=/home/altf4/benchmarks/hartford-compact-ppo-20261101/project-loop-v2/.local/direct-standard/target-4bedacf19687-capped256/release/libpatchwork_direct_standard.so
export MPLCONFIGDIR=/home/altf4/benchmarks/hartford-compact-ppo-20261101/mpl-cache
export PATCHWORK_ARENA_POLL_INFERENCE=1
export PATCHWORK_STANDARD_PLAYBACK=direct
cd /home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1
/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-launch-v1/verify_prepared.py
exec /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python -u /home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1/scripts/guard-hartford-run.py --root /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-guard-v1 --command /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-launch-v1/command.json --cwd /home/altf4/benchmarks/hartford-identity-ppo-20261102/project-v1 --min-available-mib 1536 --memory-grace-seconds 5 --keep-windows-awake --powershell /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
