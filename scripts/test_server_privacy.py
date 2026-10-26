"""Actual binary lifecycle tests. Every image/path is synthetic.

The server keeps its session data on the mounted private volume (a per-session
folder that a normal stop removes); these tests check it only through HTTP and
skip when that volume is not mounted. Legacy `<project>/.imagine` data is not
disposable history: the server must leave it untouched.
"""
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.request
import urllib.error
import zlib

ROOT = Path(__file__).resolve().parent.parent

def png():
    def chunk(name, value):
        return struct.pack('!I', len(value)) + name + value + struct.pack('!I', zlib.crc32(name + value))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 1, 1, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\x00\x11\x22\x33')) + chunk(b'IEND', b'')

class Server:
    def __init__(self, project, args):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        self.project = project
        env = dict(os.environ)
        env.pop('IMAGINE_COORDINATOR_SOCKET', None)
        env['HELTO_PRIVACY_SESSION_DIR'] = str(project / 'unused-helto-session')
        binary = os.environ.get('IMAGINE_TEST_SERVER', str(ROOT / 'target/debug/imagine-server'))
        self.process = subprocess.Popen(
            [binary, '--project', str(project), '--config', str(ROOT / 'config'),
             '--profiles', str(project / 'profiles'), '--bind', f'127.0.0.1:{self.port}',
             '--backend', 'native', *args],
            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def ready(self):
        until = time.monotonic() + 15
        while time.monotonic() < until:
            if self.process.poll() is not None:
                raise AssertionError('Synthetic server exited unexpectedly')
            try:
                return self.request('/v1/session')
            except (OSError, urllib.error.URLError):
                time.sleep(.05)
        raise AssertionError('Synthetic server did not start without terminal input')
    def request(self, path, body=None):
        r = urllib.request.Request(f'http://127.0.0.1:{self.port}{path}', data=body,
            headers={'Content-Type': 'image/png'})
        with urllib.request.urlopen(r, timeout=3) as response:
            return json.load(response)
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            return self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise AssertionError('Synthetic server did not stop') from None

def private_volume_mounted():
    with open('/proc/self/mountinfo') as mounts:
        return any(line.split()[4] == '/run/media/veracrypt1' and '/dev/mapper/veracrypt1' in line
                   for line in mounts)

@unittest.skipUnless(private_volume_mounted(), 'the private volume is not mounted')
class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='imagine-privacy-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'profiles').mkdir()
        (self.root / 'profiles/synthetic.yaml').write_text(json.dumps({
            'schema_version': 1, 'id': 'synthetic', 'version': 1, 'name': 'Synthetic',
            'description': 'Synthetic fixture', 'defaults': {'model': 'qwen-image-2.1-int8'}}))
    def start(self, *args):
        server = Server(self.root, args)
        self.addCleanup(server.stop)
        return server
    def legacy(self):
        """Old project-local data, with the files a former server owned."""
        data = self.root / '.imagine'
        (data / 'media').mkdir(parents=True, exist_ok=True)
        (data / 'media/stale').write_bytes(b'synthetic old data')
        for suffix in ('', '-wal', '-shm', '-journal'):
            (data / ('imagine.sqlite' + suffix)).write_bytes(b'synthetic old database')
        return self.snapshot()
    def snapshot(self):
        data = self.root / '.imagine'
        return {str(p.relative_to(data)): p.read_bytes() for p in sorted(data.rglob('*')) if p.is_file()}
    def check_session_lifecycle(self, *args):
        legacy = self.legacy()
        server = self.start(*args)
        session = server.ready()
        # Session data is on the encrypted private volume, also with the legacy flag.
        self.assertTrue(session['encrypted'])
        with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/v1/session', timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(json.load(response)['id'], session['id'])
        asset = server.request('/v1/assets', png())
        with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/v1/assets/' + asset['id'], timeout=3) as response:
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(response.read(), png())
        # Session data never lands in the project folder, and old data there stays as it was.
        self.assertEqual(self.snapshot(), legacy)
        self.assertEqual(server.stop(), 0)
        self.assertEqual(self.snapshot(), legacy)
        # A new process is a new session: the old session's uploads are gone.
        restarted = self.start()
        self.assertNotEqual(restarted.ready()['id'], session['id'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            restarted.request('/v1/assets/' + asset['id'])
        self.assertEqual(error.exception.code, 404)
        error.exception.close()
        self.assertEqual(restarted.stop(), 0)
        self.assertEqual(self.snapshot(), legacy)
    def test_session_data_is_per_process_and_legacy_project_data_untouched(self):
        self.check_session_lifecycle()
    def test_legacy_no_encryption_flag_remains_compatible(self):
        self.check_session_lifecycle('--no-encryption')

if __name__ == '__main__': unittest.main()
