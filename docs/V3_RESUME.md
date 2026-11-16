# Continuing incomplete Kaggle jobs

Resume continues the same scientific job, including its original job JSON,
source revision, seed, optimizer/scaler state, best weights and validation history.
The dispatch manifest's run ID identifies the new session; it does not replace the
original training job's run ID. Training revision remains
`92d778c356331f5fb3d8f9e66f7388b33821a1ad` for the first v3 cohort.

Each new private kernel attaches one immutable input owned by its execution
account: either an original kernel output pinned as `owner/slug/1`, or a private
checkpoint dataset pinned as `owner/dataset/1` when moving to another account.
Only the verified checkpoint and unchanged job JSON are copied to that dataset;
no source video, account credentials or pool files are uploaded.

The record's `resume` object contains exactly one of `source_kernel` or
`source_dataset`, plus `checkpoint_sha256`, `job_sha256`, `epoch`, `next_batch` and
`optimizer_updates`. Use the actual recorded hashes and counters. The launcher
requires a version-pinned reference on the execution owner and rechecks GPU quota.
The worker rejects missing, duplicated, altered or completed inputs. It copies
the original job JSON unchanged and passes the read-only mounted checkpoint to
the pinned runner's `--resume` path. The runner also checks data fingerprints.

After preparing and checking the manifest:

```powershell
python scripts/kaggle_v3.py --action push --pool-file D:/private/pool.json --manifest outputs_v3/resume_session2_manifest.json --only-jobs spatial_qp_seed2027
python scripts/kaggle_v3.py --action status --pool-file D:/private/pool.json --manifest outputs_v3/resume_session2_manifest.json
```

Confirm `V3_RESUME_INPUT_VERIFIED` and a subsequent `V3_STEP` whose successful
update count exceeds the checkpoint counter before bulk dispatch. Input
verification alone does not prove that training has restored and advanced the
model/optimizer/scaler. Source/output kernels are preserved; session outputs use
new kernels and remain distinct from the first-session checkpoints.

An additional 7.8-hour session is a resumable budget, not a convergence claim.
Only completed/converged jobs with validation-selected weights qualify for final
held-out evaluation under the protocol.
