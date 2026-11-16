# V3 execution report — 2026-11-16 (Asia/Bangkok)

Repository: https://github.com/munnn01/t_scope_v3.
Training revision: `04d5aa3de9114fcf64691caddbad7a40ac3347c8`.
Later documentation commits do not change the source pinned in the workers.

## Validation

- Linux CI, including real FFmpeg/x265:149 tests passed,1 skipped,299 subtests
  passed. https://github.com/munnn01/t_scope_v3/actions/runs/43733966333.
- GPU canary completed on Tesla T4 with Python3.11.14 and torch2.5.1+cu124.
  Two actual optimizer updates;14.20997seconds for the training loop;
  peak allocated GPU memory2,966,776,320bytes. Environment setup and data
  hashing are excluded; this is not a throughput or accuracy benchmark.
- Model, optimizer and scaler were restored from the saved checkpoint. The
  downloaded checkpoint's SHA256 matched and its model tensors were finite.
  See [machine-readable canary evidence](V3_GPU_CANARY.json).

## Production launch

Run ID: `v3-1116-train`. Current quota inspection found18 eligible accounts and
5 accounts below the minimum. All18 jobs were submitted successfully; the
post-dispatch status check observed **18 RUNNING,0 queued,0 launch errors**.
The live stream of `spatial_seed2026` additionally confirmed an optimizer update
(`epoch=1, batch=2/783, update=1, loss=6.09150`).

| Variant | Independent seeds |
|---|---|
| spatial | 2026,2027,2028 |
| spatial_qp | 2026,2027,2028 |
| temporal | 2026,2027,2028 |
| temporal_qp | 2026,2027,2028 |
| temporal_qp_proxy | 2026,2027,2028 |
| scope_paper_adapted | 2026,2027,2028 |

The private local launch manifest is `outputs_v3/train_20261116_manifest.json`.
It retains the account/kernel assignments, worker hashes, pinned source and job
statuses. Credentials remain in the external pool file.

All jobs use the full common train split, fixed512x512/clip4/batch2/accumulation2,
36maximum epochs, min_epochs12, patience10 and a7.8-hour session checkpoint
deadline. Hitting that deadline produces resumable incomplete work, not convergence.
Resume/evaluation must use the job's exact source revision and configuration.

These are observed launch states, not completed training or a final v3 benchmark.
Collect checkpoints and complete the paired held-out evaluation before reporting AP/BDBR.
