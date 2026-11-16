# V3 validation

Environment: Windows, Python3.11.14, torch2.5.1, torchvision0.20.1, NumPy2.4.6,
SciPy1.17.1, Pillow11.1.0. CUDA was unavailable locally.

Command: `python -B -m pytest -q -p no:cacheprovider` with OMP/MKL threads2.
Result: **145 passed, 2 skipped, 288 subtests passed**. Skips: real FFmpeg codec
integration (tools absent from local PATH) and the optional cached-pretrained
detector test. All source files compile on Python3.11; `git diff --check` passes.
The publication file set was compared against every actual pool credential; no
credential was present. Pool and runtime directories are excluded from git.

An independent read-only review found and the implementation addressed:

- Spatial QG received default temporal dQP penalties: spatial variants now zero
  those coefficients; a behavioral loss test catches recurrence of this confound.
- Validation bypassed the session deadline: save a pre-validation checkpoint,
  check deadlines during validation and preserve an explicit resumable phase.
- Legacy Kaggle keys could inherit ambient bearer credentials: the launcher now
  fails closed for legacy keys and checks retained explicit modern-token identity.
- Canary success previously did not prove checkpoint readability: safe readback
  now validates job identity and restores model, optimizer and scaler before success.

Model/reference gradient, frame-independence, memory retention, matched factorial
settings, deterministic crops, atomic checkpoint, resume identity, raw AP/rate
aggregation, PCHIP/cubic sensitivity and secret-free worker boundaries are tested.

These checks establish local code contracts, not v3 AP/FPS or production GPU
memory. Before bulk launch, the production canary must run on Kaggle CUDA with
the real pretrained teacher and dynamic real-codec ROI forward/backward. Its
machine-readable output is the authoritative execution evidence. A submitted or
queued job is not a trained model or a completed benchmark.

GitHub Actions also verified commit `a0de10d4fc5ac2c6117177b56e38696398acb819`
on Linux with real FFmpeg/x265: **142 passed, 1 skipped, 279 subtests passed**.
Run: https://github.com/munnn01/t_scope_v3/actions/runs/43732326300.

The first Kaggle launch exposed SDK2.2.5 wrapping a nonexistent-kernel HTTP403
as ValueError. The launcher now independently enumerates every page of verified
owned kernels before treating that ambiguous response as absence. Existing
kernels, HTTP401, unknown ownership and incomplete listings still block upload.
Four additional regression tests cover this production behavior.

Kaggle's owned-kernel listing also returned redacted sentinel records with id0,
empty ref/author/slug and exact title `[Private Notebook]`. The launcher now skips
only that exact service placeholder while still traversing all pages and rejecting
other malformed entries. The final focused launcher suite has24 passing tests
and25 passing subtests; an existing target on a later page still prevents upload.

The first accepted GPU canary exposed Kaggle's current Python3.13 image, which
cannot install torchvision0.20.1. Workers now provision isolated Python3.11.14 with
uv0.9.9 and execute both job creation and training with that interpreter. The
focused launcher suite has25 passing tests and30 passing subtests, including
failures at every provisioning stage and matched train/canary configurations.

## Final code verification

The initial training revision was `04d5aa3de9114fcf64691caddbad7a40ac3347c8`.
GitHub Actions on Linux, including real FFmpeg/x265, passed **149 tests**, with
**1 skipped** and **299 passing subtests**:
https://github.com/munnn01/t_scope_v3/actions/runs/43733966333.

## Initial two-update GPU canary

The production canary completed on Tesla T4 using Python3.11.14,
torch2.5.1+cu124, torchvision0.20.1+cu124 and FFmpeg6.1.1. It performed two
optimizer updates with the real pretrained detector and the real x265 training path.
The measured training loop took14.20997 seconds; peak allocated GPU memory was
2,966,776,320 bytes. This timing excludes environment setup, weight download and
data hashing, and is not a full throughput benchmark.

The checkpoint was safely loaded and model/optimizer/scaler restored before the
worker reported success. The downloaded file was independently hash-checked and
loaded locally; every model tensor was finite. SHA256:
`4d646743c80cc057b8b5888a7be691f747f735063d43347535166f57971656ba`.
Machine-readable evidence: [V3_GPU_CANARY.json](V3_GPU_CANARY.json).

The canary passed the model's predicted QG offsets into the codec, but did not
record whether nonzero frame-varying ROI was activated in those first two updates.
That activation is not claimed by the GPU timing report. Nonzero frame-varying
ROI encoding/decoding is separately covered by the real-codec CI integration test.

This validates execution, not final detection accuracy or convergence. All
production jobs must use the same pinned source revision and complete validation
before held-out evaluation.

## AMP recovery verification

The initial two-update canary did not establish sustained numerical stability:
all 18 subsequently launched jobs failed at a guard that raised before AMP could
skip an overflowed update and lower its loss scale. The repaired training revision
is `92d778c356331f5fb3d8f9e66f7388b33821a1ad`.

The local suite was rerun after the repair: **151 passed, 2 skipped, 299 subtests
passed**. Linux CI, including real FFmpeg/x265, passed **152 tests, 1 skipped,
299 subtests**: [recovery CI](https://github.com/munnn01/t_scope_v3/actions/runs/43780555476).
Three added regression tests exercise actual GradScaler overflow, scale backoff
and recovery, fatal gradients without AMP, and clipping large finite gradients.

All six variant canaries subsequently completed 128 successful optimizer updates
on Tesla T4. They each exercised recoverable overflow and continued with reduced
loss scales. Checkpoints restored successfully. See the
[recovery report](V3_AMP_RECOVERY.md) and its machine-readable evidence for the
verified GPU runs. These checks establish numerical recovery and execution;
completed training and paired AP/BDBR evaluation remain separate requirements.
