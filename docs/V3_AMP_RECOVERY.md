# V3 AMP recovery — 2026-11-16

The original 18-job launch entered RUNNING. The first failure inspection found 17
jobs in ERROR; a complete log audit subsequently confirmed all 18 with the same exception:
`Nonfinite model gradient; refusing an invalid optimizer step`.
For seed 2026, spatial stopped after 31 updates and temporal-QG after 61; the
paper-adapted reference encountered it before its first update. The two-update
canary validated execution but did not establish sustained numerical stability.

## Root cause and reproduction

The training loop unscaled gradients, checked their norm and raised before
`GradScaler.step/update` could skip an overflowed update and lower its loss scale.
Using the actual training-boundary statements with real CPU GradScaler reproduced
the exception from a finite loss whose scaled gradient overflowed. The standard
GradScaler control preserved the parameter and backed the scale from 65536 to 32768.

## Repair

- Delegate overflow skips and loss-scale backoff to GradScaler; do not update
  weights or count a successful optimizer update for an overflow.
- Compute clipping norms in float64 so large finite fp32 gradients do not overflow
  the norm reduction and become incorrectly zeroed.
- Clear gradients after either outcome; checkpoint skip counts and current scale.
- Keep nonfinite unscaled/FP32 gradients fatal. Sixteen consecutive AMP overflows
  stop with a diagnostic checkpoint instead of silently skipping indefinitely.
- Require 128 successful updates for the recovery canary, rather than 2.

Three new behavioral regression tests cover overflow/backoff/recovery, fatal
unscaled NaNs and clipping a large finite gradient. Full local suite: 151 passed,
2 environment skips, 299 passing subtests. Independent review also exercised real
CPU GradScaler+AdamW and confirmed unchanged weights/moments on skipped updates.

GPU recovery must pass the 128-update soak before bulk redispatch. Restart the
matched matrix from the original seeds under one new pinned revision; do not
mix the failed revision's checkpoints into the repaired comparison. Original
logs and manifests remain available locally for audit.

## Verified recovery soaks

The repaired temporal-QG seed 2026 canary completed 128 successful optimizer updates
on Tesla T4. It encountered 4 recoverable overflows and reduced its scale from 65536
to 4096 while continuing; about 785 seconds of training-loop execution passed the original
failure point. Model/optimizer/scaler checkpoint restoration succeeded, and the
downloaded file's hash and finite model tensors were checked independently.
See [recorded GPU evidence](V3_AMP_SOAK.json).

The same 128-update check then passed on every other variant, all at seed 2026
and revision `92d778c356331f5fb3d8f9e66f7388b33821a1ad`:

| Variant | Successful updates | AMP-skipped windows | Final loss scale | Loop seconds |
|---|---:|---:|---:|---:|
| spatial | 128 | 3 | 8192 | 808.02 |
| spatial_qp | 128 | 2 | 16384 | 730.48 |
| temporal | 128 | 3 | 8192 | 821.96 |
| temporal_qp | 128 | 4 | 4096 | 785.47 |
| temporal_qp_proxy | 128 | 3 | 8192 | 778.77 |
| scope_paper_adapted | 128 | 2 | 16384 | 744.16 |

These are Tesla T4 execution checks with the pretrained training detector and
the canary's real-codec path. Loop timings exclude setup and data hashing and
do not establish comparative throughput. In particular, the proxy variant's
canary uses real codec execution; its production training retains the prescribed
proxy-only forward path.

All six downloaded checkpoint hashes, model/optimizer tensor finiteness and
model/optimizer/scaler restoration are independently checked locally. The source
hashes match the exact pinned Git blobs, and train/validation fingerprints agree
across all six runs. See [full recovery evidence](V3_AMP_SOAK_MATRIX.json).
These checks establish numerical recovery across the variant matrix, not
convergence or held-out AP/BDBR.

Successful update counters exclude AMP-skipped windows. Skips remain explicit
diagnostics in logs and checkpoints; they are never reported as optimizer updates.
RUNNING is a platform state, not proof of sustained training: inspect advancing
`V3_STEP` records and distinguish normal `V3_AMP_OVERFLOW` backoff from a terminal
exception or `numerical_failure` checkpoint.
