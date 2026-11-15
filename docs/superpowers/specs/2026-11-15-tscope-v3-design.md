# T-SCOPE v3: research and execution design

## Intent and authorization

The user requested an implemented v3, publication to
https://github.com/munnn01/t_scope_v3, and execution on multiple Kaggle accounts.
The user selected code plus experiment protocol rather than a documentation-only
roadmap. Existing v2 weights and benchmark remain explicitly historical artifacts.

## Research claim

Study the interaction between a motion-compensated temporal pre-editor and
semantic QG quantization offsets under real x265. SCOPE's conditioning, FiLM,
object/background decomposition and teacher supervision are prior work, not new
contributions. A paper-derived SCOPE reference is an adaptation to MOT17/x265,
not an assertion of reproducing the authors' COCO results or original weights.

## Approaches considered

1. Documentation and repair only: useful but would not deliver the implemented
   experiment platform the user selected.
2. Replace the entire architecture with an untested new network: adds confounders
   before the existing temporal and QG claims are isolated.
3. Keep the working v2 backbone and add a versioned, matched v3 experiment path:
   selected from the user's request and preceding review. Preserve v2 interfaces;
   give v3 its own configuration, checkpoint, job, and result schema.

## Concrete experiment

Six variants: `spatial`, `spatial_qp`, `temporal`, `temporal_qp`,
`temporal_qp_proxy`, `scope_paper_adapted`. The first four are a 2x2 factorial
on the same TemporalSCOPE backbone. The temporal factor is a package of motion
compensation, recurrence, EMA and temporal losses. The proxy control differs from
`temporal_qp` only in codec training. The paper reference uses the image-only
H/4 architecture, direct attention blend, bounded 0.09/0.03 residuals and is
reported separately from the factorial comparison.

Use seeds 2026, 2027 and 2028, the same full training split and identical optimizer,
crop, severity and stopping rules for each job. One account runs one complete
variant/seed replicate; no cross-seed model soups and no account-dependent train
partition. Train teacher: pretrained Faster R-CNN only. YOLOv8s is evaluation-only
and must load successfully when requested. Do not label transfer to a teacher
used for training as zero-shot. Only x265 family 0 is trained and claimed.

Train: MOT17-02/04/05/10. Validation: MOT17-09. Held-out evaluation:
MOT17-11/13, paired frame IDs, no overlap, official tail exclusions retained.
Initial distributed work is training, not a final v3 benchmark. Each job saves
incremental best weights, configuration/source/dependency hashes and epoch history.
An execution time budget saves a resumable checkpoint and labels incomplete work
as incomplete; it must not confer convergence or final-ready status.

## Interfaces and ownership

- `models/scope_reference.py`: `ScopePaperAdapted` with the same forward result
  keys as TemporalSCOPE, with zero QG offsets and no recurrent outputs.
- `scripts/v3_protocol.py`: `VARIANTS`, `SEEDS`, `variant_settings(name)`,
  `build_model(name, **kwargs)` and `build_job_matrix(seeds=SEEDS)`; main runner
  consumes settings `disable_temporal`, `max_delta_qp`, `use_real_x265_ste`,
  `lambda_temp_bg`, `lambda_temp_obj`, `model_kind`.
- `scripts/rd_metrics.py`: `bd_rate(..., interpolation='pchip')` and
  `compare_bd_rates(...)`, returning common-interval/monotonicity diagnostics;
  no extrapolation, no silently repaired nonmonotone curves. Cubic is sensitivity
  analysis; historical v2 cubic numbers stay unchanged.
- `scripts/run_v3.py`: strict real-data training and paired evaluation, with
  standalone JSON job configuration and schema version 3 checkpoints.
- `scripts/kaggle_v3.py`: quota inspection, dry-run manifest, revision-pinned
  dispatch, status and download; credentials never enter generated notebooks,
  outputs, git or exception logs. Dispatch only accounts with sufficient quota.

## Reproducibility and publication

Python 3.11 support; pinned tested core torch/torchvision pair; optional Kaggle and
YOLO dependencies. CPU offline test suite and real-codec integration test when
ffmpeg/ffprobe exist. Publish source and inherited history to the requested empty
repository, preserving attribution and labeling v2 results. Do not upload papers,
MOT images, pool credentials or untracked local secrets. Keep the existing origin
remote; publish only to the explicit v3 destination.

## Success criteria

1. Existing tests pass on the local Python 3.11 environment; regression fixes for
   the dataset syntax and random generator call are demonstrated.
2. Model/reference shape, gradient, temporal invariance and QG factorial contracts
   pass; matched job configurations share data, seed and budget.
3. PCHIP constant-saving, non-overlap, nonmonotone, invalid input and sensitivity
   tests pass; v2 numeric artifacts remain unchanged.
4. v3 training/evaluation and Kaggle worker commands run in dry-run/offline tests;
   fail closed on synthetic data, missing teacher, missing real codec or a changed
   source revision.
5. Remote files/commit are verified. Multi-account training is dispatched and its
   accepted/running state is observed. GPU quota or platform failures are reported
   precisely with unaffected work completed.

## Limits

No new accuracy or speed claim without measured v3 results. A reference adapted
from the paper is not an official SCOPE reproduction. A 2x2 factorial separates
the temporal package from QG, not every temporal submodule. New datasets, longer
GOPs and other codecs remain measured follow-up experiments.
