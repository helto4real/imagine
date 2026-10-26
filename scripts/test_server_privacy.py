"""Actual binary lifecycle tests. Every image/path is synthetic."""
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
    def stale(self):
        data = self.root / '.imagine'
        (data / 'media').mkdir(parents=True, exist_ok=True)
        (data / 'media/stale').write_bytes(b'synthetic old data')
        for suffix in ('', '-wal', '-shm', '-journal'):
            (data / ('imagine.sqlite' + suffix)).write_bytes(b'synthetic old database')
    def assert_purged(self):
        data = self.root / '.imagine'
        self.assertFalse((data / 'media/stale').exists())
        for suffix in ('', '-wal', '-shm', '-journal'):
            self.assertFalse((data / ('imagine.sqlite' + suffix)).exists())
    def check_session_lifecycle(self, *args):
        self.stale()
        server = self.start(*args)
        session = server.ready()
        self.assert_purged()
        self.assertFalse(session['encrypted'])
        self.assertFalse((self.root / '.imagine/privacy-keystore.json').exists())
        self.assertFalse((self.root / '.imagine/access.json').exists())
        with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/v1/session', timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(json.load(response)['id'], session['id'])
        asset = server.request('/v1/assets', png())
        path = self.root / '.imagine/media' / session['id'] / asset['id']
        # Uploads are session-owned RAM data; only explicit project Save persists media.
        self.assertFalse(path.exists())
        with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/v1/assets/' + asset['id'], timeout=3) as response:
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(response.read(), png())
        self.assertEqual(server.stop(), 0)
        self.assertFalse(path.exists())
        restarted = self.start()
        self.assertNotEqual(restarted.ready()['id'], session['id'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            restarted.request('/v1/assets/' + asset['id'])
        self.assertEqual(error.exception.code, 404)
        error.exception.close()
        self.assertEqual(restarted.stop(), 0)
    def test_default_start_wipes_before_api_and_cleans_on_stop(self):
        self.check_session_lifecycle()
    def test_legacy_no_encryption_flag_remains_compatible(self):
        self.check_session_lifecycle('--no-encryption')

if __name__ == '__main__': unittest.main()
