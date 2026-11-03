# Inert entry-trigger mandatory-loop recovery

Repair commit: 4f64940e46e0181bf963c0de0f6f5a9a51a926b5.
Issue: https://github.com/altf4/rustmage/issues/571
PR: https://github.com/altf4/rustmage/pull/572 (reviewed head pinned for runtime).

The Solitary Cell / Perilous Snare / Pinnacle Starcage loop also returns
Spring-Loaded Sawblades, whose entry-damage trigger has no legal target. The
old conservative proof rejected this fourth object and redundant ordering
choice, so game 42776 reached 6,000 decisions instead of a mandatory rules draw.
The new proof admits at most one tightly constrained auxiliary, checks actual
printed packet identity and target availability, commutes only the provably
irrelevant ordering on a disposable clone, and still requires normalized full
state equality and absence of legal escape choices.

Both independent nonauthor reviews approved the final code/test hashes. All
10 targeted tests and all 3,024 native tests passed; full guarded release
validation passed (424.7 seconds, 3.2 GiB peak under the 6 GiB limit). The first
funding test was corrected to distinguish 3W sorcery craft from the actual 6WW
instant-speed Starcage escape; earlier failed logs are preserved.

Production stopped at checkpoint 42. Under run.lock, the full original round
43, 999 successful outcomes, original failed result/trajectory, report,
manifest and guard were archived. Read-only reconstruction of the failed
prefix matched original inputs and captured the trigger cycle; it produced no
production retry result. A new source/runtime snapshot and separate libraries
were built using the previous verified Raytracer toolchains. Initial staging
and missing-PATH build failures were corrected before any production mutation;
original build logs remain in separate control directories.

On the repaired runtime, exactly the failed scheduled identity was retried,
with unchanged seed 412610552776 and checkpoints 42/39. It ended in a natural
rules draw after 196 decisions (turn 9), matching all 20 recorded learner-prefix
samples. Model weights and all old checkpoints stayed unchanged. The runtime
amendment and original artifact hashes are preserved under
run-v1/decision-limit-investigation-v1/runtime-amendment.json.

Ordinary resume under a fresh guard reused all 1,000 round-43 outcomes. Its
collection phase took 0.11 seconds and created no inference collectors, proving
successful games were not replayed. Both PPO epochs and the 200 whole-game
holdouts completed; checkpoint 43 committed. The complete original archive,
all 43 pre-repair checkpoints and the 999 retained summaries passed equality
and hash checks after resume. Training continues; evaluations remain disabled.

The on-demand monitor was updated to send only needed metadata over SSH, since
the full recovery manifests exceed the OS per-argument size limit. The hourly
heartbeat now names the current source/runtime and preserves this recovery's
provenance for the final completion audit.
