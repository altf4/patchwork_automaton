# Preserved identity-input experiment

This repository record preserves the completed 100K-game identity training arm
and its separately authorized 11,200-game comparison. The trained agent has
300K total collection-game exposure, including its original 200K parent.

- [Training protocol and completion](../training-100k/README.md)
- [Complete per-round report](../training-100k/report.json.gz)
- [Audited comparison results and figures](../post-training-comparisons/README.md)
- [Frozen source archive](frozen-source.tar.gz) and [SHA256 manifest](source-manifest.json)

The archive contains all 387 source files in the completed run's launch
manifest, verified byte-for-byte against those hashes. It includes the runtime
engine pin and Cargo inputs from `project-trigger-v2`. The editable package
outside this archive retains its own development build configuration; it must
not be mistaken for the exact recovered runtime. Runtime revision
`4f64940e46e0181bf963c0de0f6f5a9a51a926b5` and feature revision `e732a4b`
are intentionally separate. External RustMage source and compiled libraries
are identified by the recorded revision and artifact hashes, not bundled here.

To inspect the frozen code, extract it into a new, empty directory and verify
every file against `source-manifest.json`. Preparation, recovery and evaluation
scripts retain historical absolute paths. Adapt paths and reserve fresh output
directories before using them for a new experiment. Do not rerun historical
adoption scripts or write into the completed experiment.

Production model weights, optimizer/RNG checkpoints, original failed attempts,
and raw run artifacts remain on Raytracer; this Git record does not back them
up. The final checkpoint is
`/home/altf4/benchmarks/hartford-identity-ppo-20261102/run-v1/checkpoints/round-0100`.
Its policy SHA256 is
`7c5290d0a6ea4ad2d5697087cb8bd1880a52e04c48b84f2c0d171541e709b845`.
The copied parent policy SHA256 is
`eb56f40c7e3519a9cbb69dbbc9d85129bec91a6bf600d13ce3d599c5ee64fef6`.
Launch manifests and completion audits retain full artifact locations and hashes.

Repeated monitor snapshots and acceptance checkpoint/trajectory binaries are
left out of Git. The original local and remote evidence remains untouched.
The compressed training report retains all 100 rounds; the comparison archive
retains all 11,200 outcome records, schedules, model hashes and audit results.
