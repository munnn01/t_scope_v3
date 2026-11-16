"""Plan and launch revision-pinned, full-data v3 Kaggle replicates.

The account pool is read only by the local launcher. Its credentials are never
copied into manifests, worker source, metadata, or error messages.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "https://github.com/munnn01/t_scope_v3.git"
DEFAULT_DATASET = "ahmedsamir1598/mot17challenge"
DEFAULT_OVERRIDES = {"epochs": 36, "min_epochs": 12, "patience": 10, "max_hours": 7.8}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def load_account_pool(path) -> list[tuple[str, str]]:
    """Read credentials, rejecting ambiguous duplicate/invalid account names."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Invalid account pool JSON") from None
    if not isinstance(data, dict):
        raise ValueError("Account pool must map usernames to credentials")
    accounts = []
    seen = set()
    for username, credential in data.items():
        if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", username.strip()):
            raise ValueError("Invalid Kaggle account name")
        username = username.strip()
        if username.casefold() in seen or not isinstance(credential, str) or not credential.strip():
            raise ValueError("Duplicate account or empty credential")
        seen.add(username.casefold())
        accounts.append((username, credential.strip()))
    return accounts


@contextlib.contextmanager
def _quiet_api():
    # Some SDK failures print request headers before raising. Keep SDK chatter
    # local and emit only our allowlisted status/error type afterwards.
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous_level)


def get_authenticated_api(username: str, credential: str):
    """Create an isolated modern-access-token client for one pool account.

    Legacy API keys are deliberately unsupported: SDK legacy clients may reread
    ambient bearer credentials when they later create their HTTP clients.
    Calls are deliberately sequential across accounts.
    """
    if not isinstance(credential, str) or not credential.startswith("KGAT_") or len(credential) <= 5:
        raise ValueError("Only KGAT_ access tokens are supported; legacy API keys are unsupported")
    sdk_path = ROOT / ".runtime" / "kaggle"
    if sdk_path.is_dir() and str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    environment = {key: os.environ.get(key) for key in
                   ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    try:
        os.environ.update(KAGGLE_USERNAME=username, KAGGLE_KEY=credential, KAGGLE_API_TOKEN=credential)
        with _quiet_api():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            configuration = api.config_values
            if (configuration.get("token") != credential or
                    configuration.get("username", "").casefold() != username.casefold()):
                raise ValueError("Verified access-token account does not match the pool entry")
        return api
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _remaining_hours(api) -> float:
    quota = api.quota_view().gpu_quota
    remaining = (quota.total_time_allowed.total_seconds() - quota.time_used.total_seconds()) / 3600.0
    if not math.isfinite(remaining):
        raise ValueError("Invalid quota")
    return max(0.0, remaining)


def inspect_accounts(accounts, *, min_gpu_hours=8.0, max_accounts=None,
                     api_factory=None) -> list[dict[str, Any]]:
    if min_gpu_hours <= 0 or not math.isfinite(min_gpu_hours):
        raise ValueError("Minimum GPU hours must be positive")
    api_factory = api_factory or get_authenticated_api
    records = []
    for username, credential in accounts[:max_accounts]:
        record = {"username": username, "quota_hours_remaining": None}
        try:
            with _quiet_api():
                api = api_factory(username, credential)
                remaining = _remaining_hours(api)
            record.update(quota_hours_remaining=remaining,
                          status="ready" if remaining >= min_gpu_hours else "quota_low")
        except Exception as error:
            record.update(status="quota_error", error_type=type(error).__name__)
        records.append(record)
    return records


def _validate_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full 40-character published commit SHA is required")


def _validate_repo(repo_url):
    parsed = urlparse(repo_url)
    if (parsed.scheme != "https" or parsed.netloc != "github.com" or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", parsed.path)):
        raise ValueError("Repository must be a public HTTPS GitHub URL without credentials")


def _assert_secret_free(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(part in str(key).lower() for part in ("token", "password", "credential", "secret", "api_key")):
                raise ValueError("Secret fields are forbidden in launch artifacts")
            _assert_secret_free(item)
    elif isinstance(value, list):
        for item in value:
            _assert_secret_free(item)


def build_manifest(jobs, quota_records, *, run_id, revision, min_gpu_hours=8.0,
                   dataset_slug=DEFAULT_DATASET, repo_url=DEFAULT_REPO,
                   canary_only=False, max_accounts=None, only_jobs=None,
                   job_overrides=None) -> dict[str, Any]:
    """Assign stable job IDs once; unavailable replicas remain explicit pending jobs."""
    _validate_revision(revision)
    _validate_repo(repo_url)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", run_id):
        raise ValueError("Run ID must contain 1-40 lowercase letters, digits or hyphens")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset_slug):
        raise ValueError("Dataset must be an owner/slug reference")
    if min_gpu_hours <= 0 or not math.isfinite(min_gpu_hours):
        raise ValueError("Minimum GPU hours must be positive")
    if max_accounts is not None and max_accounts < 1:
        raise ValueError("Maximum accounts must be positive")
    usernames = [record["username"].casefold() for record in quota_records]
    if len(usernames) != len(set(usernames)):
        raise ValueError("Duplicate quota account")
    selected_jobs = set(only_jobs) if only_jobs is not None else None
    known_ids = [job["job_id"] for job in jobs]
    if len(known_ids) != len(set(known_ids)):
        raise ValueError("Duplicate job ID")
    if selected_jobs is not None and selected_jobs - set(known_ids):
        raise ValueError("Unknown requested job ID")
    quotas = [{"username": record["username"],
               "quota_hours_remaining": record.get("quota_hours_remaining"),
               "status": record["status"],
               **({"error_type": record["error_type"]} if "error_type" in record else {})}
              for record in quota_records]
    available = [record for record in quotas
                 if record["status"] == "ready" and
                 record.get("quota_hours_remaining") is not None and
                 record["quota_hours_remaining"] >= min_gpu_hours]
    if max_accounts is not None:
        available = available[:max_accounts]
    if canary_only:
        available = available[:1]
    overrides = dict(DEFAULT_OVERRIDES)
    overrides.update(job_overrides or {})
    manifest = {"schema_version": 3, "run_id": run_id, "revision": revision,
                "repo_url": repo_url, "dataset_slug": dataset_slug,
                "min_gpu_hours": min_gpu_hours, "mode": "canary" if canary_only else "train",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "job_overrides": overrides, "accounts": quotas, "jobs": []}
    account_index = 0
    for job in jobs:
        if selected_jobs is not None and job["job_id"] not in selected_jobs:
            continue
        if job["job_id"] != f'{job["variant"]}_seed{job["seed"]}' or not re.fullmatch(r"[a-z0-9_]+", job["job_id"]):
            raise ValueError("Job ID must match variant and seed")
        record = {key: deepcopy(job[key]) for key in ("job_id", "variant", "seed", "settings")}
        record.update(username=None, kernel_id=None, quota_hours_remaining=None, status="pending")
        if account_index < len(available):
            account = available[account_index]
            account_index += 1
            identity = f'{run_id}:{manifest["mode"]}:{job["job_id"]}'
            digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
            short_variant = job["variant"].replace("_", "-")[:8].rstrip("-")
            slug = f'tscope-v3-{run_id[:12].rstrip("-")}-{manifest["mode"]}-{short_variant}-{digest}'
            record.update(username=account["username"], kernel_id=f'{account["username"]}/{slug}',
                          quota_hours_remaining=account["quota_hours_remaining"], status="planned")
        manifest["jobs"].append(record)
    _assert_secret_free(manifest)
    return manifest


def save_manifest(manifest, path):
    _assert_secret_free(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if manifest.get("schema_version") != 3:
        raise ValueError("Expected schema-3 launch manifest")
    _validate_revision(manifest["revision"])
    _validate_repo(manifest["repo_url"])
    _assert_secret_free(manifest)
    return manifest


def _validate_resume_spec(record, manifest):
    spec = record.get("resume")
    if spec is None:
        return
    common = {"checkpoint_sha256", "job_sha256", "epoch", "next_batch", "optimizer_updates"}
    if (not isinstance(spec, dict) or manifest["mode"] != "train" or
            set(spec) not in (common | {"source_kernel"}, common | {"source_dataset"})):
        raise ValueError("Resume requires an exact production-training input specification")
    source = spec.get("source_kernel", spec.get("source_dataset"))
    if (not isinstance(source, str) or
            not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*", source) or
            source.split("/")[0].casefold() != str(record["username"]).casefold()):
        raise ValueError("Resume source must be version-pinned and owned by the execution account")
    for key in ("checkpoint_sha256", "job_sha256"):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError("Resume requires SHA256 identities for checkpoint and job")
    for key in ("epoch", "next_batch", "optimizer_updates"):
        if type(spec[key]) is not int or spec[key] < (1 if key == "epoch" else 0):
            raise ValueError("Invalid expected resume position")


def build_worker_script(record, manifest) -> str:
    """Generate an importable standalone worker without copying local credentials."""
    _validate_revision(manifest["revision"])
    _validate_repo(manifest["repo_url"])
    _validate_resume_spec(record, manifest)
    payload = {"revision": manifest["revision"], "repo_url": manifest["repo_url"],
               "run_id": manifest["run_id"],
               "mode": manifest["mode"], "variant": record["variant"],
               "seed": record["seed"], "job_id": record["job_id"],
               "job_overrides": manifest["job_overrides"]}
    if record.get("resume") is not None:
        payload["resume"] = deepcopy(record["resume"])
    _assert_secret_free(payload)
    return '''"""Revision-pinned T-SCOPE v3 worker; credentials remain on the launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CONFIG = json.loads(''' + repr(json.dumps(payload, allow_nan=False)) + ''')

JOB_CREATION = """import json
from pathlib import Path
import sys
from scripts.prepare_kaggle_datasets import find_mounted_kaggle_dataset
from scripts.run_v3 import make_job

config = json.loads(sys.argv[1])
work_root = Path(sys.argv[2])
data_root = find_mounted_kaggle_dataset()
if data_root is None or not Path(data_root).is_dir():
    raise RuntimeError("Mounted real MOT17 training dataset is required")
if config.get("resume"):
    import shutil
    import torch
    from scripts.run_v3 import sha256_file, validate_resume
    spec = config["resume"]
    inputs = Path(sys.argv[3])
    checkpoints = list(inputs.rglob(config["job_id"] + ".last.pt"))
    jobs = list(inputs.rglob(config["job_id"] + ".json"))
    if len(checkpoints) != 1 or len(jobs) != 1 or checkpoints[0].parent != jobs[0].parent:
        raise ValueError("Resume input is missing or ambiguous")
    checkpoint, job_file = checkpoints[0], jobs[0]
    if sha256_file(checkpoint) != spec["checkpoint_sha256"] or sha256_file(job_file) != spec["job_sha256"]:
        raise ValueError("Resume checkpoint or job hash mismatch")
    job = json.loads(job_file.read_text(encoding="utf-8"))
    if any(job.get(key) != config[key] for key in ("job_id", "variant", "seed")) or job.get("source_revision") != config["revision"]:
        raise ValueError("Resume job identity differs from the dispatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_resume(state, job)
    if state["status"] != "time_budget_exhausted" or any(state[key] != spec[key] for key in ("epoch", "next_batch", "optimizer_updates")):
        raise ValueError("Checkpoint is not the expected incomplete resume position")
    shutil.copyfile(job_file, work_root / job_file.name)
    (work_root / (config["job_id"] + ".resume-path.json")).write_text(json.dumps(str(checkpoint)), encoding="utf-8")
    print("V3_RESUME_INPUT_VERIFIED " + json.dumps({"job_id": job["job_id"], **spec}), flush=True)
else:
    overrides = dict(config["job_overrides"], source_revision=config["revision"], run_id=config["run_id"])
    job = make_job(config["variant"], config["seed"], **overrides)
    (work_root / (config["job_id"] + ".json")).write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
(work_root / (config["job_id"] + ".data-root.json")).write_text(json.dumps(str(data_root)), encoding="utf-8")
"""

def main(work_root=Path("/kaggle/working"), input_root=Path("/kaggle/input")):
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    repo = Path(tempfile.gettempdir()) / ("tscope-v3-" + CONFIG["revision"])
    if not repo.exists():
        subprocess.run(["git", "clone", CONFIG["repo_url"], str(repo)], check=True)
    subprocess.run(["git", "checkout", "--detach", CONFIG["revision"]], cwd=repo, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != CONFIG["revision"]:
        raise RuntimeError("Pinned worker source revision mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("Pinned worker checkout has local modifications")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.9.9"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "python", "install", "3.11.14"], check=True)
    runtime = Path(tempfile.gettempdir()) / ("tscope-v3-env-" + CONFIG["revision"][:12])
    subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.11.14", str(runtime)], check=True)
    managed_python = str(runtime / "bin" / "python")
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", managed_python,
                    "-r", str(repo / "requirements.txt")], check=True)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Real x265 requires ffmpeg and ffprobe")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=1",
                    "-frames:v", "1", "-c:v", "libx265", "-f", "null", "-"], check=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([managed_python, "-c", JOB_CREATION, json.dumps(CONFIG), str(work_root), str(input_root)],
                   cwd=repo, env=env, check=True)
    data_root = Path(json.loads((work_root / (CONFIG["job_id"] + ".data-root.json")).read_text(encoding="utf-8")))
    if not data_root.is_dir():
        raise RuntimeError("Mounted real MOT17 training dataset is required")
    job_file = work_root / (CONFIG["job_id"] + ".json")
    command = [managed_python, str(repo / "scripts" / "run_v3.py"), "--mode", CONFIG["mode"],
               "--job-file", str(job_file), "--data-root", str(data_root), "--output-dir", str(work_root)]
    if CONFIG.get("resume"):
        resume_path = json.loads((work_root / (CONFIG["job_id"] + ".resume-path.json")).read_text(encoding="utf-8"))
        command += ["--resume", resume_path]
    subprocess.run(command, cwd=repo, env=env, check=True)

if __name__ == "__main__":
    main()
'''


def _pool_map(accounts):
    users = [username.casefold() for username, _ in accounts]
    if len(users) != len(set(users)):
        raise ValueError("Duplicate account")
    return dict(accounts)


def _selection(manifest, only_jobs):
    selected = set(only_jobs) if only_jobs is not None else None
    known = {record["job_id"] for record in manifest["jobs"]}
    if selected is not None and selected - known:
        raise ValueError("Unknown requested job ID")
    return [record for record in manifest["jobs"] if selected is None or record["job_id"] in selected]


def _http_status_codes(error):
    """Read status codes through SDK wrappers without formatting exception text."""
    pending, seen, codes = [error], set(), set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for code in (getattr(current, "status", None), getattr(current, "status_code", None),
                     getattr(getattr(current, "response", None), "status_code", None)):
            if type(code) is int:
                codes.add(code)
        pending.extend((current.__cause__, current.__context__))
    return codes


def _missing_kernel(error):
    return _http_status_codes(error) == {404}


def _owned_kernel_absent(api, record, credential):
    """Confirm absence after ambiguous 403 using every verified owned-kernel page.

    The SDK converts some nonexistent-kernel 403s to ValueError. This fallback
    never infers absence from the status failure itself: it independently lists
    this authenticated owner's kernels and fails closed on incomplete results.
    """
    username = record["username"]
    configuration = getattr(api, "config_values", {})
    reference = record["kernel_id"].split("/")
    if (not isinstance(configuration, dict) or
            not isinstance(configuration.get("username"), str) or
            configuration["username"].casefold() != username.casefold() or
            not isinstance(credential, str) or not credential.startswith("KGAT_") or
            configuration.get("token") != credential or
            len(reference) != 2 or not reference[1] or
            reference[0].casefold() != username.casefold()):
        raise ValueError("Kernel ownership and authenticated account must match")
    page_token, seen_tokens = None, set()
    while True:
        response = api.kernels_list_with_response(
            mine=True, user=username, page_size=100, page_token=page_token)
        if (response is None or not isinstance(getattr(response, "kernels", None), (list, tuple)) or
                not hasattr(response, "next_page_token")):
            raise ValueError("Owned-kernel listing did not return a complete page")
        for kernel in response.kernels:
            item_reference = getattr(kernel, "ref", None)
            # Kaggle redacts some inaccessible notebooks to this exact sentinel,
            # even on an owner-filtered page. It carries no usable identity.
            if (type(getattr(kernel, "id", None)) is int and kernel.id == 0 and
                    item_reference == "" and getattr(kernel, "title", None) == "[Private Notebook]" and
                    getattr(kernel, "author", None) == "" and getattr(kernel, "slug", None) == ""):
                continue
            parts = item_reference.split("/") if isinstance(item_reference, str) else []
            if len(parts) != 2 or not parts[1] or parts[0].casefold() != username.casefold():
                raise ValueError("Owned-kernel listing returned an unverifiable reference")
            if parts[1] == reference[1]:
                return False
        next_token = response.next_page_token
        if next_token is None or next_token == "":
            return True
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise ValueError("Owned-kernel listing pagination is invalid or repeated")
        seen_tokens.add(next_token)
        page_token = next_token


def _status_string(response):
    raw = getattr(response, "status", "unknown")
    return str(getattr(raw, "name", raw)).rsplit(".", 1)[-1].lower()


def push_manifest(manifest, accounts, *, api_factory=None, only_jobs=None, persist=None):
    """Push only new kernel IDs; every existing kernel is left intact."""
    _validate_revision(manifest["revision"])
    api_factory = api_factory or get_authenticated_api
    credentials = _pool_map(accounts)
    for record in _selection(manifest, only_jobs):
        if record["username"] is None or record["status"] in ("submitted", "running", "queued", "complete", "downloaded"):
            continue
        try:
            _validate_resume_spec(record, manifest)
            if record["username"] not in credentials:
                record["status"] = "account_missing"
                continue
            with _quiet_api():
                api = api_factory(record["username"], credentials[record["username"]])
                remaining = _remaining_hours(api)
                record["quota_hours_remaining"] = remaining
                if remaining < manifest["min_gpu_hours"]:
                    record["status"] = "quota_low"
                    continue
                if record.get("resume", {}).get("source_kernel") and _status_string(api.kernels_status(record["resume"]["source_kernel"])) != "complete":
                    raise ValueError("Resume source kernel has not completed")
                try:
                    api.kernels_status(record["kernel_id"])
                except Exception as error:
                    if not _missing_kernel(error):
                        if _http_status_codes(error) != {403}:
                            raise
                        if not _owned_kernel_absent(api, record, credentials[record["username"]]):
                            record["status"] = "existing_kernel"
                            continue
                else:
                    record["status"] = "existing_kernel"
                    continue
                with tempfile.TemporaryDirectory(prefix="tscope_v3_") as directory:
                    path = Path(directory)
                    code = build_worker_script(record, manifest)
                    (path / "worker.py").write_text(code, encoding="utf-8")
                    metadata = {"id": record["kernel_id"], "title": record["kernel_id"].split("/")[1],
                                "code_file": "worker.py", "language": "python", "kernel_type": "script",
                                "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
                                "enable_internet": "true", "dataset_sources": [manifest["dataset_slug"]] + ([record["resume"]["source_dataset"]] if record.get("resume", {}).get("source_dataset") else []),
                                "competition_sources": [], "kernel_sources": [record["resume"]["source_kernel"]] if record.get("resume", {}).get("source_kernel") else [], "model_sources": []}
                    (path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    response = api.kernels_push(str(path))
                    if getattr(response, "error", ""):
                        record.update(status="push_rejected", error_type="KagglePushRejected")
                    else:
                        record.update(status="submitted", worker_sha256=hashlib.sha256(code.encode()).hexdigest())
        except Exception as error:
            record.update(status="push_error", error_type=type(error).__name__)
        finally:
            if persist:
                persist(manifest)
    return manifest


def status_manifest(manifest, accounts, *, api_factory=None, only_jobs=None, persist=None):
    api_factory = api_factory or get_authenticated_api
    credentials = _pool_map(accounts)
    for record in _selection(manifest, only_jobs):
        if record["username"] is None:
            continue
        try:
            with _quiet_api():
                api = api_factory(record["username"], credentials[record["username"]])
                status = _status_string(api.kernels_status(record["kernel_id"]))
            # Unknown SDK strings might contain service diagnostics. Emit only
            # expected state labels and retain an explicit unknown state.
            record["status"] = status if status in {
                "running", "queued", "complete", "error", "cancelled", "canceled", "pending"} else "unknown"
            record.pop("error_type", None)
        except Exception as error:
            record.update(status="not_found" if _missing_kernel(error) else "status_error", error_type=type(error).__name__)
        finally:
            if persist:
                persist(manifest)
    return manifest


def pull_manifest(manifest, accounts, output_dir, *, api_factory=None, only_jobs=None,
                  persist=None, downloader=None):
    api_factory = api_factory or get_authenticated_api
    if downloader is None:
        from scripts.kaggle_output_download import download_kernel_outputs
        downloader = download_kernel_outputs
    credentials = _pool_map(accounts)
    root = Path(output_dir).resolve() / manifest["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    for record in _selection(manifest, only_jobs):
        if record["username"] is None:
            continue
        try:
            with _quiet_api():
                api = api_factory(record["username"], credentials[record["username"]])
                state = _status_string(api.kernels_status(record["kernel_id"]))
                if state != "complete":
                    record["download_status"] = "not_complete"
                    continue
                # A fresh directory keeps retries from silently mixing versions.
                destination = Path(tempfile.mkdtemp(prefix=record["job_id"]+"_", dir=root))
                downloader(api, record["kernel_id"], destination)
            record.update(download_status="downloaded", output_directory=str(destination))
        except Exception as error:
            record.update(download_status="download_error", error_type=type(error).__name__)
        finally:
            if persist:
                persist(manifest)
    return manifest


def _snapshot(manifest):
    print(json.dumps(manifest, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("verify", "plan", "push", "status", "pull"), required=True)
    parser.add_argument("--pool-file", required=True)
    parser.add_argument("--manifest", default="outputs_v3/launch_manifest.json")
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--min-gpu-hours", type=float, default=8.0)
    parser.add_argument("--max-accounts", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--only-jobs", nargs="+")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="outputs_v3")
    args = parser.parse_args(argv)
    try:
        accounts = load_account_pool(args.pool_file)
        if args.max_accounts is not None and args.max_accounts < 1:
            raise ValueError("Maximum accounts must be positive")
        if args.action == "verify":
            snapshot = {"schema_version": 3, "accounts": inspect_accounts(accounts,
                min_gpu_hours=args.min_gpu_hours, max_accounts=args.max_accounts)}
            _snapshot(snapshot)
            return 0
        manifest_path = Path(args.manifest)
        if args.action == "plan" or (args.action == "push" and not manifest_path.exists()):
            _validate_revision(args.revision)
            from scripts.v3_protocol import build_job_matrix
            quotas = inspect_accounts(accounts, min_gpu_hours=args.min_gpu_hours)
            run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:6]
            manifest = build_manifest(build_job_matrix(seeds=tuple(args.seeds)), quotas,
                run_id=run_id, revision=args.revision, min_gpu_hours=args.min_gpu_hours,
                repo_url=args.repo_url, dataset_slug=args.dataset_slug, canary_only=args.canary_only,
                max_accounts=args.max_accounts, only_jobs=args.only_jobs)
            if manifest_path.exists():
                raise ValueError("Plan manifest already exists; use a new manifest path")
            save_manifest(manifest, manifest_path)
        else:
            manifest = load_manifest(manifest_path)
            if args.revision and args.revision != manifest["revision"]:
                raise ValueError("CLI revision differs from saved manifest")
            if args.run_id and args.run_id != manifest["run_id"]:
                raise ValueError("CLI run ID differs from saved manifest")
            if args.canary_only and manifest["mode"] != "canary":
                raise ValueError("Canary flag differs from saved manifest")
        persist = lambda value: save_manifest(value, manifest_path)
        if args.action == "push":
            push_manifest(manifest, accounts, only_jobs=args.only_jobs, persist=persist)
        elif args.action == "status":
            status_manifest(manifest, accounts, only_jobs=args.only_jobs, persist=persist)
        elif args.action == "pull":
            pull_manifest(manifest, accounts, args.output_dir, only_jobs=args.only_jobs, persist=persist)
        _snapshot(manifest)
        return 0
    except Exception as error:
        # Never format SDK/pool exception text: it can include bearer tokens.
        print(json.dumps({"status": "launcher_error", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
