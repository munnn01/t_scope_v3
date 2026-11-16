# V3 resume-session execution record

Observed UTC: 2026-11-16T16:12:28.653644+00:00

All 12 incomplete jobs were submitted to new private notebooks. The observed states were 9 RUNNING and 3 QUEUED. Successful optimizer counters advanced beyond the original checkpoints in 9 jobs. Queued jobs have not yet demonstrated restoration or training progress.

Training uses the original job JSON and source revision `92d778c356331f5fb3d8f9e66f7388b33821a1ad`, retaining seeds, optimizer/scaler state, validation history and stopping rules. The launcher revision is `38e1a5abe84843e0de95b640e892a3a01ae789e0`.

One job uses a private version-1 checkpoint dataset on the account selected by the user. Its privacy and readiness were verified through the API. The other 11 jobs attach version-1 outputs owned by their execution accounts. The old notebooks/checkpoints remain intact.

| Job | Observed state | Checkpoint update | Observed update | Restored progress verified |
|---|---|---:|---:|---|
| `spatial_seed2026` | running | 4536 | 4548 | True |
| `spatial_seed2027` | running | 4537 | 4543 | True |
| `spatial_seed2028` | running | 4552 | 4553 | True |
| `spatial_qp_seed2026` | running | 4394 | 4396 | True |
| `spatial_qp_seed2027` | running | 4386 | 4516 | True |
| `spatial_qp_seed2028` | running | 4511 | 4520 | True |
| `temporal_seed2026` | queued | 4514 | not observed | False |
| `temporal_seed2027` | running | 4347 | 4377 | True |
| `temporal_seed2028` | running | 4583 | 4603 | True |
| `temporal_qp_seed2026` | running | 4206 | 4229 | True |
| `temporal_qp_seed2027` | queued | 4407 | not observed | False |
| `temporal_qp_seed2028` | queued | 4403 | not observed | False |

Local validation: 158 passed, 2 skipped, 309 subtests. Linux CI with real x265: 159 passed, 1 skipped, 309 subtests. [CI run](https://github.com/munnn01/t_scope_v3/actions/runs/43901101468). Independent review found no actionable issue in the resume path.

Input verification requires exact checkpoint/job hashes, identity and counters. Advancing `V3_STEP` records provide runtime evidence that the pinned runner restored and continued the training state. [Machine-readable evidence](V3_RESUME_SESSION2.json).

These records establish resume execution only. All jobs retain the 7.8-hour session budget. Complete remaining training and paired held-out evaluation before reporting AP/BDBR or a v3-versus-reference claim.
