# V3 AMP recovery — 2026-11-16

The original18-job launch entered RUNNING, but later inspection found17 jobs in
ERROR. All six variant representatives reported the same exception:
`Nonfinite model gradient; refusing an invalid optimizer step`.
For seed2026, spatial stopped after31 updates and temporal-QG after61; the
paper-adapted reference encountered it before its first update. The two-update
canary validated execution but did not establish sustained numerical stability.

## Root cause and reproduction

The training loop unscaled gradients, checked their norm and raised before
`GradScaler.step/update` could skip an overflowed update and lower its loss scale.
Using the actual training-boundary statements with real CPU GradScaler reproduced
the exception from a finite loss whose scaled gradient overflowed. The standard
GradScaler control preserved the parameter and backed the scale from65536 to32768.

## Repair

- Delegate overflow skips and loss-scale backoff to GradScaler; do not update
  weights or count a successful optimizer update for an overflow.
- Compute clipping norms in float64 so large finite fp32 gradients do not overflow
  the norm reduction and become incorrectly zeroed.
- Clear gradients after either outcome; checkpoint skip counts and current scale.
- Keep nonfinite unscaled/FP32 gradients fatal. Sixteen consecutive AMP overflows
  stop with a diagnostic checkpoint instead of silently skipping indefinitely.
- Require128 successful updates for the recovery canary, rather than2.

Three new behavioral regression tests cover overflow/backoff/recovery, fatal
unscaled NaNs and clipping a large finite gradient. Full local suite:151 passed,
2 environment skips,299 passing subtests. Independent review also exercised real
CPU GradScaler+AdamW and confirmed unchanged weights/moments on skipped updates.

GPU recovery must pass the128-update soak before bulk redispatch. Restart the
matched matrix from the original seeds under one new pinned revision; do not
mix the failed revision's checkpoints into the repaired comparison. Original
logs and manifests remain available locally for audit.
