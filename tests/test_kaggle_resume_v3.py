"""Resume the original job and optimizer position from immutable kernel output."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import kaggle_v3 as launcher
from scripts.run_v3 import make_job
from scripts.v3_protocol import build_job_matrix

REVISION = 'a' * 40


class ResumeWorkerTests(unittest.TestCase):
    def fixture(self, root):
        inputs, output = root / 'input', root / 'output'
        source = inputs / 'original-kernel'
        source.mkdir(parents=True)
        job = make_job('spatial_qp', 2027, source_revision=REVISION, run_id='original-run')
        job_path = source / (job['job_id'] + '.json')
        job_path.write_text(json.dumps(job, indent=2), encoding='utf-8')
        checkpoint = source / (job['job_id'] + '.last.pt')
        state = {'schema_version': 3, 'job': job, 'status': 'time_budget_exhausted',
                 'model_state': {'weight': torch.ones(2)}, 'optimizer_state': {},
                 'scaler_state': {}, 'epoch': 12, 'next_batch': 158, 'history': [],
                 'optimizer_updates': 4386}
        torch.save(state, checkpoint)
        manifest = launcher.build_manifest([r for r in build_job_matrix(seeds=(2027,)) if r['variant']=='spatial_qp'],
            [{'username': 'ready', 'quota_hours_remaining': 12, 'status': 'ready'}],
            run_id='resume-session2', revision=REVISION)
        record = manifest['jobs'][0]
        record['resume'] = {'source_kernel': 'ready/original-kernel/1',
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'job_sha256': hashlib.sha256(job_path.read_bytes()).hexdigest(),
            'epoch': 12, 'next_batch': 158, 'optimizer_updates': 4386}
        return inputs, output, job, checkpoint, job_path, state, manifest, record

    def execute_worker(self, manifest, record, inputs, output):
        namespace = {'__name__': 'worker_import'}
        exec(compile(launcher.build_worker_script(record, manifest), 'worker.py', 'exec'), namespace)
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            if argv[1:2] == ['-c']:
                with patch.object(sys, 'argv', ['-c', *argv[3:]]):
                    exec(compile(argv[2], 'job_creation.py', 'exec'), {'__name__': '__main__'})
            return SimpleNamespace(returncode=0)
        with patch('subprocess.run', side_effect=run), \
             patch('subprocess.check_output', side_effect=lambda argv, **kw: REVISION if argv[-1]=='HEAD' else ''), \
             patch('shutil.which', return_value='ffmpeg'), \
             patch('scripts.prepare_kaggle_datasets.find_mounted_kaggle_dataset', return_value=inputs):
            namespace['main'](work_root=output, input_root=inputs)
        return calls

    def test_resume_preserves_exact_job_and_passes_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, output, job, checkpoint, job_path, state, manifest, record = self.fixture(Path(directory))
            calls = self.execute_worker(manifest, record, inputs, output)
            restored_job = output / job_path.name
            self.assertEqual(restored_job.read_bytes(), job_path.read_bytes())
            self.assertEqual(json.loads(restored_job.read_text()), state['job'])
            command = calls[-1]
            self.assertEqual(command[command.index('--resume')+1], str(checkpoint))
            self.assertEqual(command[command.index('--mode')+1], 'train')
            self.assertIn(['git', 'checkout', '--detach', REVISION], calls)
            self.assertFalse((output / checkpoint.name).exists(), 'Source checkpoint must remain read-only')

    def test_tampered_checkpoint_or_job_fails_before_training(self):
        for target in ('checkpoint', 'job'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                inputs, output, job, checkpoint, job_path, state, manifest, record = self.fixture(Path(directory))
                path = checkpoint if target == 'checkpoint' else job_path
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'hash'):
                    self.execute_worker(manifest, record, inputs, output)

    def test_wrong_resume_position_or_completed_checkpoint_fails(self):
        for mutation in ({'epoch': 11}, {'next_batch': 160}, {'optimizer_updates': 1}, {'status': 'converged'}):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                inputs, output, job, checkpoint, job_path, state, manifest, record = self.fixture(Path(directory))
                state.update(mutation)
                torch.save(state, checkpoint)
                record['resume']['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.execute_worker(manifest, record, inputs, output)

    def test_missing_or_ambiguous_mount_fails(self):
        for kind in ('missing', 'duplicate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                inputs, output, job, checkpoint, job_path, state, manifest, record = self.fixture(Path(directory))
                if kind == 'missing':
                    checkpoint.unlink()
                else:
                    duplicate = inputs / 'duplicate'
                    duplicate.mkdir()
                    (duplicate / checkpoint.name).write_bytes(checkpoint.read_bytes())
                with self.assertRaises(ValueError):
                    self.execute_worker(manifest, record, inputs, output)

    def test_resume_metadata_pins_original_kernel_and_rechecks_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            *_, manifest, record = self.fixture(Path(directory))
            uploads = []
            class Api:
                def quota_view(self):
                    from datetime import timedelta
                    return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=timedelta(hours=30), time_used=timedelta(hours=18)))
                def kernels_status(self, ref):
                    if ref == record['resume']['source_kernel']:
                        return SimpleNamespace(status='complete')
                    error = RuntimeError('missing')
                    error.status = 404
                    raise error
                def kernels_push(self, path):
                    uploads.append(json.loads((Path(path)/'kernel-metadata.json').read_text()))
                    return SimpleNamespace(error='')
            launcher.push_manifest(manifest, [('ready','KGAT_unit_test')], api_factory=lambda *args: Api())
            self.assertEqual(record['status'], 'submitted')
            self.assertEqual(uploads[0]['kernel_sources'], ['ready/original-kernel/1'])

    def test_cross_owner_or_unpinned_source_is_rejected(self):
        for source in ('another/original-kernel/1', 'ready/original-kernel'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                *_, manifest, record = self.fixture(Path(directory))
                record['resume']['source_kernel'] = source
                with self.assertRaises(ValueError):
                    launcher.build_worker_script(record, manifest)

    def test_private_dataset_resume_preserves_job_and_pins_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, output, job, checkpoint, job_path, state, manifest, record = self.fixture(Path(directory))
            record['resume'].pop('source_kernel')
            record['resume']['source_dataset'] = 'ready/resume-checkpoint/1'
            calls = self.execute_worker(manifest, record, inputs, output)
            self.assertEqual(calls[-1][calls[-1].index('--resume')+1], str(checkpoint))
            self.assertEqual((output/job_path.name).read_bytes(), job_path.read_bytes())
            uploads=[]
            class Api:
                def quota_view(self):
                    from datetime import timedelta
                    return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=timedelta(hours=30), time_used=timedelta(hours=0)))
                def kernels_status(self, ref):
                    error=RuntimeError('missing'); error.status=404; raise error
                def kernels_push(self, path):
                    uploads.append(json.loads((Path(path)/'kernel-metadata.json').read_text()))
                    return SimpleNamespace(error='')
            launcher.push_manifest(manifest,[('ready','KGAT_unit_test')],api_factory=lambda *args:Api())
            self.assertEqual(record['status'],'submitted')
            self.assertEqual(uploads[0]['kernel_sources'],[])
            self.assertEqual(uploads[0]['dataset_sources'],[manifest['dataset_slug'],'ready/resume-checkpoint/1'])


if __name__ == '__main__':
    unittest.main()
