"""Offline execution contracts for the v3 multi-account launcher."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import contextlib
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REVISION = "a" * 40
SECRET = "do-not-disclose-test-credential"


def jobs():
    return [
        {"job_id": "spatial_seed2026", "variant": "spatial", "seed": 2026,
         "settings": {"disable_temporal": True, "use_real_x265_ste": True}},
        {"job_id": "temporal_qp_seed2026", "variant": "temporal_qp", "seed": 2026,
         "settings": {"disable_temporal": False, "use_real_x265_ste": True}},
    ]


class FakeApi:
    def __init__(self, hours=12, status="missing", fail=None):
        self.hours, self.status, self.fail = hours, status, fail
        self.uploads = []

    def quota_view(self):
        if self.fail:
            print(SECRET)
            raise RuntimeError(SECRET)
        return SimpleNamespace(gpu_quota=SimpleNamespace(
            time_used=timedelta(hours=30-self.hours),
            total_time_allowed=timedelta(hours=30)))

    def kernels_status(self, ref):
        if self.status == "missing":
            error = RuntimeError("404")
            error.status = 404
            raise error
        return SimpleNamespace(status=self.status)

    def kernels_push(self, directory):
        path = Path(directory)
        metadata = json.loads((path / "kernel-metadata.json").read_text())
        code = (path / metadata["code_file"]).read_text()
        self.uploads.append((metadata, code))
        return SimpleNamespace(url="https://www.kaggle.com/code/ready/kernel", error="")


class ForbiddenStatusApi(FakeApi):
    """Model SDK ValueError wrapping HTTPError and paginated owned notebooks."""
    def __init__(self, pages, code=403):
        super().__init__()
        self.pages, self.code = pages, code
        self.config_values = {"username": "ready", "token": "KGAT_test_access_token"}
        self.list_calls = []

    def kernels_status(self, ref):
        error = RuntimeError(SECRET)
        error.response = SimpleNamespace(status_code=self.code)
        try:
            raise error
        except RuntimeError:
            raise ValueError("SDK obscures the status response")

    def kernels_list_with_response(self, *, mine, user, page_size, page_token):
        if (mine, user, page_size) != (True, "ready", 100):
            raise AssertionError("Listing must be scoped to verified owned kernels")
        self.list_calls.append(page_token)
        page = self.pages[page_token]
        if isinstance(page, Exception):
            raise page
        return page


class KaggleV3Tests(unittest.TestCase):
    def launcher(self):
        self.assertIsNotNone(importlib.util.find_spec("scripts.kaggle_v3"),
                             "v3 Kaggle launcher is not implemented")
        from scripts import kaggle_v3
        return kaggle_v3

    def manifest(self, launcher, **overrides):
        return launcher.build_manifest(jobs(), [
            {"username": "low", "quota_hours_remaining": 2, "status": "quota_low"},
            {"username": "ready", "quota_hours_remaining": 12, "status": "ready"},
        ], run_id="unit-run", revision=REVISION, **overrides)

    def test_assignment_keeps_job_identity_and_pending_replicates(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        first, second = manifest["jobs"]
        self.assertEqual((first["job_id"], first["username"], first["status"]),
                         ("spatial_seed2026", "ready", "planned"))
        self.assertEqual((second["job_id"], second["username"], second["status"]),
                         ("temporal_qp_seed2026", None, "pending"))
        self.assertIn("unit-run", first["kernel_id"])
        canary = self.manifest(launcher, canary_only=True)
        self.assertNotEqual(canary["jobs"][0]["kernel_id"], first["kernel_id"])
        self.assertEqual(manifest["schema_version"], 3)

    def test_rejects_invalid_revision_duplicate_accounts_and_jobs(self):
        launcher = self.launcher()
        with self.assertRaises(ValueError):
            launcher.build_manifest(jobs(), [], run_id="unit", revision="main")
        with self.assertRaises(ValueError):
            launcher.build_manifest(jobs()+jobs()[:1], [], run_id="unit", revision=REVISION)
        duplicate = [{"username": "same", "quota_hours_remaining": 12, "status": "ready"}] * 2
        with self.assertRaises(ValueError):
            launcher.build_manifest(jobs(), duplicate, run_id="unit", revision=REVISION)
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "pool.json"
            pool.write_text('{"same":"first", "same":"second"}')
            with self.assertRaises(ValueError):
                launcher.load_account_pool(pool)

    def test_long_run_ids_have_short_unique_kernel_slugs(self):
        launcher = self.launcher()
        matrix = [
            {"job_id": f"{variant}_seed{seed}", "variant": variant, "seed": seed, "settings": {}}
            for variant in ("spatial", "temporal_qp", "scope_paper_adapted")
            for seed in (2026, 2027, 2028)
        ]
        quotas = [{"username": f"ready{index}", "quota_hours_remaining": 12, "status": "ready"}
                  for index in range(9)]
        manifest = launcher.build_manifest(matrix, quotas, run_id="a"*40, revision=REVISION)
        slugs = [record["kernel_id"].split("/")[1] for record in manifest["jobs"]]
        self.assertEqual(len(set(slugs)), 9)
        for slug in slugs:
            self.assertLessEqual(len(slug), 50)
            self.assertRegex(slug, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
        self.assertEqual([record["job_id"] for record in manifest["jobs"]],
                         [job["job_id"] for job in matrix])

    def test_account_errors_do_not_escape_into_output_or_manifest(self):
        launcher = self.launcher()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            quotas = launcher.inspect_accounts([("bad", SECRET), ("ready", SECRET)],
                api_factory=lambda user, token: FakeApi(fail=user == "bad"))
        self.assertEqual(quotas[0]["status"], "quota_error")
        self.assertEqual(quotas[1]["status"], "ready")
        manifest = launcher.build_manifest(jobs(), quotas, run_id="unit", revision=REVISION)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            launcher.save_manifest(manifest, target)
            serialized = target.read_text()
        self.assertNotIn(SECRET, output.getvalue()+serialized)
        self.assertNotIn("token", serialized.lower())

    def test_sdk_logging_cannot_print_a_credential(self):
        launcher = self.launcher()
        output = io.StringIO()
        logger = logging.getLogger("tscope_test_sdk")
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        class LoggingApi(FakeApi):
            def quota_view(self):
                logger.error(SECRET)
                raise RuntimeError(SECRET)
        try:
            quotas = launcher.inspect_accounts([("ready", SECRET)],
                api_factory=lambda user, token: LoggingApi())
        finally:
            logger.removeHandler(handler)
        self.assertEqual(quotas[0]["status"], "quota_error")
        self.assertNotIn(SECRET, output.getvalue())

    def test_legacy_credentials_are_rejected_before_api_or_environment_access(self):
        launcher = self.launcher()
        class NeverApi:
            def __init__(self):
                raise AssertionError("Legacy credentials must not reach the SDK")
        module = SimpleNamespace(KaggleApi=NeverApi)
        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": module}), \
             patch.dict(os.environ, {"KAGGLE_USERNAME": "ambient-user", "KAGGLE_API_TOKEN": "KGAT_ambient"}), \
             patch.dict(os.environ) as environment:
            before = dict(environment)
            with self.assertRaisesRegex(ValueError, "legacy"):
                launcher.get_authenticated_api("ready", SECRET)
            self.assertEqual(dict(environment), before)

    def test_modern_client_retains_explicit_token_after_environment_restoration(self):
        launcher = self.launcher()
        supplied = "KGAT_"+SECRET
        class ExplicitApi:
            def authenticate(self):
                self.config_values = {"username": os.environ["KAGGLE_USERNAME"],
                                      "token": os.environ["KAGGLE_API_TOKEN"]}
        module = SimpleNamespace(KaggleApi=ExplicitApi)
        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": module}), \
             patch.dict(os.environ, {"KAGGLE_USERNAME": "ambient-user", "KAGGLE_API_TOKEN": "KGAT_ambient"}):
            api = launcher.get_authenticated_api("ready", supplied)
            self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "KGAT_ambient")
            self.assertEqual(os.environ["KAGGLE_USERNAME"], "ambient-user")
            self.assertEqual(api.config_values, {"username": "ready", "token": supplied})

    def test_token_authentication_fallback_or_wrong_account_fails_closed(self):
        launcher = self.launcher()
        supplied = "KGAT_"+SECRET
        for actual in ({"username": "ready", "token": None},
                       {"username": "another-user", "token": supplied}):
            with self.subTest(config=actual):
                class WrongApi:
                    def authenticate(self):
                        self.config_values = actual
                module = SimpleNamespace(KaggleApi=WrongApi)
                with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": module}):
                    with self.assertRaises(ValueError):
                        launcher.get_authenticated_api("ready", supplied)

    def test_push_refuses_active_or_unverifiable_kernel_without_upload(self):
        launcher = self.launcher()
        for state in ("running", "queued", "KernelWorkerStatus.RUNNING", "complete"):
            with self.subTest(state=state):
                api = FakeApi(status=state)
                manifest = self.manifest(launcher)
                launcher.push_manifest(manifest, [("ready", SECRET)],
                    api_factory=lambda user, token: api)
                self.assertEqual(api.uploads, [])
                self.assertEqual(manifest["jobs"][0]["status"], "existing_kernel")

    def test_push_preserves_identity_on_failure_and_rechecks_quota(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        before_id = manifest["jobs"][0]["kernel_id"]
        api = FakeApi(hours=1)
        launcher.push_manifest(manifest, [("ready", SECRET)], api_factory=lambda user, token: api)
        self.assertEqual(manifest["jobs"][0]["status"], "quota_low")
        self.assertEqual(manifest["jobs"][0]["kernel_id"], before_id)
        self.assertIsNone(manifest["jobs"][1]["username"])
        self.assertEqual(api.uploads, [])

    def test_worker_executes_pinned_checkout_and_strict_runner_argv(self):
        launcher = self.launcher()
        for canary_only, mode in ((False, "train"), (True, "canary")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                manifest = self.manifest(launcher, canary_only=canary_only)
                record = manifest["jobs"][0]
                source = launcher.build_worker_script(record, manifest)
                compiled = compile(source, "worker.py", "exec")
                namespace = {"__name__": "worker_import"}
                exec(compiled, namespace)
                calls = []
                def run(argv, **kwargs):
                    calls.append((argv, kwargs))
                    return SimpleNamespace(returncode=0)
                def output(argv, **kwargs):
                    calls.append((argv, kwargs))
                    return REVISION+"\n" if argv[-1] == "HEAD" else ""
                made_jobs = []
                def make_job(variant, seed, **kwargs):
                    made_jobs.append((variant, seed, kwargs))
                    return {"schema_version": 3, "job_id": "spatial_seed2026"}
                modules = {
                    "scripts.run_v3": SimpleNamespace(make_job=make_job),
                    "scripts.prepare_kaggle_datasets": SimpleNamespace(
                        find_mounted_kaggle_dataset=lambda: Path(directory)),
                }
                with patch.dict(sys.modules, modules), patch("subprocess.run", run), \
                     patch("subprocess.check_output", output), patch("shutil.which", return_value="ffmpeg"):
                    namespace["main"](work_root=Path(directory))
                command = calls[-1][0]
                self.assertEqual(command[command.index("--mode")+1], mode)
                self.assertEqual(command[command.index("--data-root")+1], directory)
                self.assertIn(["git", "checkout", "--detach", REVISION], [x[0] for x in calls])
                self.assertEqual(made_jobs[0][0:2], ("spatial", 2026))
                self.assertEqual(made_jobs[0][2]["epochs"], 36)
                self.assertEqual(made_jobs[0][2]["max_hours"], 7.8)
                self.assertEqual(made_jobs[0][2]["source_revision"], REVISION)
                self.assertEqual(made_jobs[0][2]["run_id"], "unit-run")
                clone = next(command for command, _ in calls if command[:2] == ["git", "clone"])
                self.assertFalse(Path(clone[-1]).is_relative_to(Path(directory)))
                self.assertNotIn(SECRET, source)

    def test_worker_revision_mismatch_fails_before_training(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        namespace = {"__name__": "worker_import"}
        exec(compile(launcher.build_worker_script(manifest["jobs"][0], manifest),
                     "worker.py", "exec"), namespace)
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
             patch("subprocess.run", side_effect=lambda argv, **kw: calls.append(argv)), \
             patch("subprocess.check_output", return_value="b"*40):
            with self.assertRaises(RuntimeError):
                namespace["main"](work_root=Path(directory))
        self.assertFalse(any("--mode" in command for command in calls))

    def test_worker_dirty_checkout_fails_before_training(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        namespace = {"__name__": "worker_import"}
        exec(compile(launcher.build_worker_script(manifest["jobs"][0], manifest),
                     "worker.py", "exec"), namespace)
        calls = []
        def output(argv, **kwargs):
            return REVISION if argv[-1] == "HEAD" else " M scripts/run_v3.py\n"
        with tempfile.TemporaryDirectory() as directory:
            modules = {
                "scripts.run_v3": SimpleNamespace(make_job=lambda *a, **kw: {}),
                "scripts.prepare_kaggle_datasets": SimpleNamespace(
                    find_mounted_kaggle_dataset=lambda: Path(directory)),
            }
            with patch.dict(sys.modules, modules), \
                 patch("subprocess.run", side_effect=lambda argv, **kw: calls.append(argv)), \
                 patch("subprocess.check_output", side_effect=output), \
                 patch("shutil.which", return_value="ffmpeg"):
                with self.assertRaises(RuntimeError):
                    namespace["main"](work_root=Path(directory))
        self.assertFalse(any("--mode" in command for command in calls))

    def test_plan_never_uploads_and_push_payload_contains_no_credentials(self):
        launcher = self.launcher()
        api = FakeApi()
        manifest = self.manifest(launcher)
        self.assertEqual(api.uploads, [])
        launcher.push_manifest(manifest, [("ready", SECRET)], api_factory=lambda user, token: api)
        self.assertEqual(manifest["jobs"][0]["status"], "submitted")
        self.assertEqual(len(api.uploads), 1)
        metadata, source = api.uploads[0]
        self.assertEqual(metadata["dataset_sources"], ["ahmedsamir1598/mot17challenge"])
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertNotIn(SECRET, json.dumps(metadata)+source+json.dumps(manifest))

    def test_sdk_http_404_allows_a_new_kernel_push(self):
        launcher = self.launcher()
        class MissingApi(FakeApi):
            def kernels_status(self, ref):
                error = RuntimeError(SECRET)
                error.response = SimpleNamespace(status_code=404)
                raise error
        api = MissingApi()
        manifest = self.manifest(launcher)
        launcher.push_manifest(manifest, [("ready", SECRET)], api_factory=lambda user, token: api)
        self.assertEqual(manifest["jobs"][0]["status"], "submitted")
        self.assertEqual(len(api.uploads), 1)

    def test_sdk_403_allows_new_kernel_only_after_all_owned_pages_are_absent(self):
        launcher = self.launcher()
        api = ForbiddenStatusApi({
            None: SimpleNamespace(kernels=[SimpleNamespace(ref="ready/unrelated-one")], next_page_token="page2"),
            "page2": SimpleNamespace(kernels=[SimpleNamespace(ref="ready/unrelated-two")], next_page_token=""),
        })
        manifest = self.manifest(launcher)
        launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
        self.assertEqual(manifest["jobs"][0]["status"], "submitted")
        self.assertEqual(api.list_calls, [None, "page2"])
        self.assertEqual(len(api.uploads), 1)

    def test_sdk_403_target_on_second_owned_page_is_never_overwritten(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        target = manifest["jobs"][0]["kernel_id"]
        api = ForbiddenStatusApi({
            None: SimpleNamespace(kernels=[SimpleNamespace(ref="ready/unrelated")], next_page_token="page2"),
            "page2": SimpleNamespace(kernels=[SimpleNamespace(ref=target)], next_page_token=""),
        })
        launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
        self.assertEqual(manifest["jobs"][0]["status"], "existing_kernel")
        self.assertEqual(api.uploads, [])

    def test_exact_private_notebook_placeholder_does_not_end_owned_pagination(self):
        launcher = self.launcher()
        placeholder = SimpleNamespace(id=0, ref="", title="[Private Notebook]", author="", slug="")
        for target_exists in (False, True):
            with self.subTest(target_exists=target_exists):
                manifest = self.manifest(launcher)
                target = manifest["jobs"][0]["kernel_id"]
                last_ref = target if target_exists else "ready/unrelated-two"
                api = ForbiddenStatusApi({
                    None: SimpleNamespace(kernels=[placeholder, SimpleNamespace(ref="ready/unrelated-one")],
                                          next_page_token="page2"),
                    "page2": SimpleNamespace(kernels=[SimpleNamespace(ref=last_ref)], next_page_token=""),
                })
                launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
                self.assertEqual(api.list_calls, [None, "page2"])
                self.assertEqual(manifest["jobs"][0]["status"], "existing_kernel" if target_exists else "submitted")
                self.assertEqual(len(api.uploads), 0 if target_exists else 1)

    def test_empty_reference_outside_exact_private_placeholder_stays_blocked(self):
        launcher = self.launcher()
        for fields in (
            {"id": 17, "ref": "", "title": "[Private Notebook]", "author": "", "slug": ""},
            {"id": 0, "ref": "", "title": "Another notebook", "author": "", "slug": ""},
            {"id": 0, "ref": "", "title": "[Private Notebook]", "author": "ready", "slug": ""},
            {"id": 0, "ref": "", "title": "[Private Notebook]", "author": "", "slug": "known-slug"},
        ):
            with self.subTest(fields=fields):
                api = ForbiddenStatusApi({None: SimpleNamespace(
                    kernels=[SimpleNamespace(**fields)], next_page_token="")})
                manifest = self.manifest(launcher)
                launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
                self.assertEqual(manifest["jobs"][0]["status"], "push_error")
                self.assertEqual(api.uploads, [])

    def test_sdk_403_failed_or_ambiguous_owned_listing_blocks_upload(self):
        launcher = self.launcher()
        for pages in (
            {None: RuntimeError(SECRET)},
            {None: None},
            {None: SimpleNamespace(kernels=None, next_page_token="")},
            {None: SimpleNamespace(kernels=[], next_page_token="loop"),
             "loop": SimpleNamespace(kernels=[], next_page_token="loop")},
        ):
            with self.subTest(pages=list(pages)):
                api = ForbiddenStatusApi(pages)
                manifest = self.manifest(launcher)
                launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
                self.assertEqual(manifest["jobs"][0]["status"], "push_error")
                self.assertEqual(api.uploads, [])
                self.assertNotIn(SECRET, json.dumps(manifest))

    def test_sdk_403_requires_verified_owner_and_401_never_uses_listing(self):
        launcher = self.launcher()
        for mode in ("wrong_owner", "wrong_token", "unknown_owner", "unauthorized", "generic_value_error"):
            with self.subTest(mode=mode):
                api = ForbiddenStatusApi({None: SimpleNamespace(kernels=[], next_page_token="")})
                manifest = self.manifest(launcher)
                if mode == "wrong_owner":
                    manifest["jobs"][0]["kernel_id"] = "someoneelse/new-slug"
                elif mode == "wrong_token":
                    api.config_values["token"] = "KGAT_another_access_token"
                elif mode == "unknown_owner":
                    api.config_values = {}
                elif mode == "unauthorized":
                    api.code = 401
                else:
                    api.kernels_status = lambda ref: (_ for _ in ()).throw(ValueError("No HTTP status"))
                launcher.push_manifest(manifest, [("ready", "KGAT_test_access_token")], api_factory=lambda *args: api)
                self.assertEqual(manifest["jobs"][0]["status"], "push_error")
                self.assertEqual(api.uploads, [])
                self.assertEqual(api.list_calls, [])

    def test_cli_plan_caps_ready_accounts_without_uploading(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "pool.json"
            pool.write_text(json.dumps({"low": SECRET, "ready": SECRET}))
            manifest_path = Path(directory) / "manifest.json"
            protocol = SimpleNamespace(build_job_matrix=lambda seeds: jobs())
            factory = lambda user, token: FakeApi(hours=1 if user == "low" else 12)
            with patch.dict(sys.modules, {"scripts.v3_protocol": protocol}), \
                 patch.object(launcher, "get_authenticated_api", side_effect=factory), \
                 patch.object(launcher, "push_manifest", side_effect=AssertionError("plan cannot upload")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = launcher.main(["--action", "plan", "--pool-file", str(pool),
                    "--manifest", str(manifest_path), "--run-id", "cli-plan",
                    "--revision", REVISION, "--max-accounts", "1"])
            self.assertEqual(result, 0)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["jobs"][0]["username"], "ready")
            self.assertIsNone(manifest["jobs"][1]["username"])
            self.assertNotIn(SECRET, manifest_path.read_text())

    def test_status_and_pull_preserve_identity_and_complete_output_bytes(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        api = FakeApi(status="KernelWorkerStatus.COMPLETE")
        before_id = manifest["jobs"][0]["kernel_id"]
        launcher.status_manifest(manifest, [("ready", SECRET)], api_factory=lambda user, token: api)
        self.assertEqual(manifest["jobs"][0]["status"], "complete")
        def download(api, kernel_id, destination):
            self.assertEqual(kernel_id, before_id)
            (destination / "checkpoint.pt").write_bytes(b"real-output-content")
        with tempfile.TemporaryDirectory() as directory:
            launcher.pull_manifest(manifest, [("ready", SECRET)], directory,
                api_factory=lambda user, token: api, downloader=download)
            record = manifest["jobs"][0]
            self.assertEqual(record["download_status"], "downloaded")
            self.assertEqual(record["kernel_id"], before_id)
            self.assertEqual((Path(record["output_directory"]) / "checkpoint.pt").read_bytes(), b"real-output-content")

    def test_worker_missing_codec_or_dataset_fails_before_runner(self):
        launcher = self.launcher()
        manifest = self.manifest(launcher)
        source = launcher.build_worker_script(manifest["jobs"][0], manifest)
        for codec_exists, data_exists in ((False, True), (True, False)):
            with self.subTest(codec=codec_exists, data=data_exists), tempfile.TemporaryDirectory() as directory:
                namespace = {"__name__": "worker_import"}
                exec(compile(source, "worker.py", "exec"), namespace)
                calls = []
                modules = {
                    "scripts.run_v3": SimpleNamespace(make_job=lambda *a, **kw: {}),
                    "scripts.prepare_kaggle_datasets": SimpleNamespace(
                        find_mounted_kaggle_dataset=lambda: Path(directory) if data_exists else None),
                }
                with patch.dict(sys.modules, modules), \
                     patch("subprocess.run", side_effect=lambda argv, **kw: calls.append(argv)), \
                     patch("subprocess.check_output", side_effect=lambda argv, **kw: REVISION if argv[-1] == "HEAD" else ""), \
                     patch("shutil.which", return_value="codec" if codec_exists else None):
                    with self.assertRaises(RuntimeError):
                        namespace["main"](work_root=Path(directory))
                self.assertFalse(any("--mode" in command for command in calls))


if __name__ == "__main__":
    unittest.main()
