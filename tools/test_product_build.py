"""CPU-only launcher regression tests with synthetic source/artifact fixtures."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import product_build as build


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='imagine-build-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', '-b', 'main', self.root], check=True)
        subprocess.run(['git', '-C', self.root, 'config', 'commit.gpgsign', 'false'], check=True)
        self.put('.gitignore', '/comfyui/\n')
        self.put('src/main.rs', 'synthetic source')
        self.put('ui/src/main.rs', 'synthetic UI')
        self.put('proxy/st-stack.zsh', '# synthetic supervisor')
        self.lock = {'schema': 1, 'product_files': [], 'related_source_locks': [],
                     'build': {'product_target': str(self.root / 'comfyui/target')}}
        self.put(build.LOCK, json.dumps(self.lock))
        subprocess.run(['git', '-C', self.root, 'add', '.'], check=True)
        subprocess.run(['git', '-C', self.root, '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)

    def put(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def fake_cargo(self, root, lock, bins, native=False):
        for binary in bins:
            self.put('comfyui/target/release/' + binary, '#!/bin/sh\nexit 0\n')

    def test_main_required_even_if_old_artifact_exists(self):
        self.put('comfyui/active-product.sh', 'exit 99')
        subprocess.run(['git', '-C', self.root, 'checkout', '-qb', 'experiment'], check=True)
        with patch.object(build, 'cargo') as cargo:
            with self.assertRaisesRegex(ValueError, 'requires local branch main'):
                build.prepare(self.root, 'ui', False)
            cargo.assert_not_called()

    def test_dirty_and_new_source_are_recorded(self):
        before = build.source_state(self.root, self.lock)
        self.put('ui/src/new.rs', 'new')
        after = build.source_state(self.root, self.lock)
        self.assertFalse(before['dirty'])
        self.assertTrue(after['dirty'])
        self.assertIn('ui/src/new.rs', after['files'])
        self.assertNotEqual(before, after)

    def test_failed_build_never_reuses_previous_binary(self):
        self.put('comfyui/target/release/imagine-ui', 'old executable')
        with patch.object(build, 'cargo', side_effect=subprocess.CalledProcessError(1, 'cargo')):
            with self.assertRaises(subprocess.CalledProcessError):
                build.prepare(self.root, 'ui', False)
        self.assertFalse((self.root / 'comfyui/main-build/releases').exists())

    def test_changed_source_aborts_publication(self):
        def changing_cargo(*args):
            self.fake_cargo(*args)
            self.put('ui/src/main.rs', 'changed during build')
        with patch.object(build, 'cargo', side_effect=changing_cargo):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                build.prepare(self.root, 'ui', False)
        self.assertEqual(list((self.root / 'comfyui/main-build/releases').iterdir()), [])

    def test_snapshot_preserves_old_inode_and_records_hashes(self):
        self.put('comfyui/active-product.sh', 'do not execute this')
        with patch.object(build, 'cargo', side_effect=self.fake_cargo):
            first = build.prepare(self.root, 'ui', False)
            second = build.prepare(self.root, 'ui', False)
        self.assertNotEqual(first, second)
        self.assertNotEqual((first / 'imagine-ui').stat().st_ino,
                            (second / 'imagine-ui').stat().st_ino)
        report = json.loads((second / 'build.json').read_text())
        self.assertEqual(report['artifacts']['imagine-ui'], build.digest(second / 'imagine-ui'))

    def test_verifier_rejects_change_with_restored_mtime(self):
        artifact = self.put('library.so', 'old')
        expected = build.digest(artifact)
        verifier = build.Verifier(self.root / 'receipt.json')
        verifier.files({str(artifact): expected})
        stamp = artifact.stat()
        artifact.write_text('new')
        os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'):
            verifier.files({str(artifact): expected})

    def test_native_core_comes_from_git_and_only_allowlisted_files_follow_main(self):
        self.put('inference/core.rs', 'frozen')
        self.put('inference/adapter.rs', 'old adapter')
        self.put('shared/base.rs', 'base')
        self.put('design/h3-stage4-evidence/selected-model-sha256.json', '{}')
        subprocess.run(['git', '-C', self.root, 'add', '.'], check=True)
        subprocess.run(['git', '-C', self.root, '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'native'], check=True)
        revision = build.git(self.root, 'rev-parse', 'HEAD').decode().strip()
        lock = {'source_base': revision, 'core_revision': revision, 'core_files': {},
                'core_patches': {}, 'product_files': ['inference/adapter.rs']}
        self.put('inference/core.rs', 'unapproved core edit')
        self.put('inference/adapter.rs', 'new adapter')
        self.put('inference/unlisted.rs', 'must not copy')
        with build.build_context(self.root) as (cache, verifier):
            source = build.native_source(self.root, lock, cache, verifier)
            self.assertEqual((source / 'inference/core.rs').read_text(), 'frozen')
            self.assertEqual((source / 'inference/adapter.rs').read_text(), 'new adapter')
            self.assertFalse((source / 'inference/unlisted.rs').exists())
            (source / 'inference/core.rs').write_text('tampered cache')
            with self.assertRaisesRegex(ValueError, 'Native source cache changed'):
                build.native_source(self.root, lock, cache, verifier)

    def test_real_protocol_matches_lock_and_mismatch_fails(self):
        lock = build.load_lock(build.ROOT)
        build.validate_protocol(build.ROOT, lock)
        lock['h3']['identity']['protocol'] += 1
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            build.validate_protocol(build.ROOT, lock)

    def test_core_patch_hash_and_product_overlap_fail_before_restoration(self):
        patch_path = self.put('native/core.patch',
            'diff --git a/inference/core.rs b/inference/core.rs\n'
            '--- a/inference/core.rs\n+++ b/inference/core.rs\n'
            '@@ -1 +1 @@\n-old\n+new\n')
        lock = dict(source_base='unused', core_revision='unused', core_files={},
                    core_patches={'native/core.patch': '0' * 64}, product_files=[])
        with build.build_context(self.root) as (cache, verifier):
            with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'):
                build.native_source(self.root, lock, cache, verifier)
            lock['core_patches']['native/core.patch'] = build.digest(patch_path)
            lock['product_files'] = ['inference/core.rs']
            with self.assertRaisesRegex(ValueError, 'overwrite core patch'):
                build.native_source(self.root, lock, cache, verifier)

    def test_help_and_explicit_backend_do_not_prepare_native(self):
        for args in (['--help'], ['--backend', 'comfy'], ['--backend', 'native'],
                     ['--rust-runtime', '/synthetic/custom.json']):
            self.assertFalse(build.needs_runtime(args))
        self.assertTrue(build.needs_runtime([]))
        self.assertTrue(build.needs_runtime(['--backend', 'rust']))
        self.assertTrue(build.needs_runtime(['--backend', 'native', '--backend', 'rust']))
        self.assertFalse(build.needs_runtime(['--backend', 'rust', '--backend', 'comfy']))
        self.assertTrue(build.needs_runtime(['--data', '--help']))
        # Path options of the server keep the default Rust runtime.
        for flag in ('--config', '--video-presets', '--private-project-root', '--mcp-root'):
            self.assertTrue(build.needs_runtime([flag, '/synthetic/path']), flag)

    def test_private_coordinator_and_cli_arguments_are_preserved(self):
        with patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/synthetic/run'}, clear=True):
            env = build.launch_environment(self.root)
        self.assertEqual(env['IMAGINE_PRIVATE_STORAGE'], '1')
        self.assertEqual(env['IMAGINE_COORDINATOR_SOCKET'], '/synthetic/run/imagine/gpu.sock')
        self.assertEqual(env['ST_PROXY_COORDINATOR_SOCKET'], env['IMAGINE_COORDINATOR_SOCKET'])
        args = ['--bind', '127.0.0.1:8087', '--data', '/synthetic/path with spaces']
        directory = self.root / 'release'
        with patch.object(build, 'ROOT', self.root), patch.object(build, 'prepare', return_value=directory), \
             patch.object(build.os, 'execve') as execute, \
             patch.object(build.sys, 'argv', ['build', 'launch', 'server', '--', *args]), \
             patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/synthetic/run'}):
            build.main()
        self.assertEqual(execute.call_args.args[1][-len(args):], args)
        self.assertIn(str(directory / 'runtime.json'), execute.call_args.args[1])

    def test_stop_does_not_build_or_require_main(self):
        with patch.object(build, 'prepare') as prepare, \
             patch.object(build.os, 'execvpe', side_effect=SystemExit) as execute, \
             patch.object(build.sys, 'argv', ['build', 'launch', 'proxy', '--', '--stop']), \
             patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/synthetic/run'}):
            with self.assertRaises(SystemExit):
                build.main()
        prepare.assert_not_called()
        self.assertEqual(execute.call_args.args[1][-1], '--stop')

    def test_explicit_package_validates_binary_and_never_executes_selector(self):
        directory = self.root / 'comfyui/old-package'
        binary = self.put('comfyui/old-package/imagine-ui', 'old UI')
        self.put('comfyui/old-package/package.json', json.dumps({
            'ui_sha256': build.digest(binary), 'private_storage': True}))
        self.put('comfyui/old-package/active-product.sh', 'exit 99')
        self.assertEqual(build.package(self.root, directory, 'ui'), (directory, True))
        binary.write_text('wrong')
        with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'):
            build.package(self.root, directory, 'ui')

    def test_release_profile_builds_with_release_flag_without_debug_info(self):
        lock = {'build': {'product_target': '/own/target', 'native_target': '/own/worker',
                          'jobs': 2, 'rustflags': '-C debuginfo=0', 'profile': 'release'}}
        self.assertEqual(build.output(lock, 'native_target'), Path('/own/worker/release'))
        env = build.build_env(lock)
        self.assertEqual(env['CARGO_PROFILE_RELEASE_DEBUG'], '0')
        with patch.object(build, 'run') as run:
            build.cargo(self.root, lock, ('imagine-ui',))
        self.assertIn('--release', run.call_args.args[0])
        lock['build']['profile'] = 'dev'
        self.assertEqual(build.output(lock, 'product_target'), Path('/own/target/debug'))
        with patch.object(build, 'run') as run:
            build.cargo(self.root, lock, ('imagine-ui',))
        self.assertNotIn('--release', run.call_args.args[0])
        lock['build']['profile'] = 'bench'
        with self.assertRaisesRegex(ValueError, 'Unsupported build profile'):
            build.output(lock, 'product_target')

if __name__ == '__main__':
    unittest.main()
