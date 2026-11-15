# Local v3 validation — 2026-11-15

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
