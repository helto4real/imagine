"""Actual binary/TTY lifecycle tests. Every credential/image/path is synthetic."""
import json
import os
from pathlib import Path
import pty
import select
import signal
import socket
import struct
import tempfile
import time
import unittest
import urllib.request
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
        self.output = b''
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            env = dict(os.environ)
            env.pop('IMAGINE_COORDINATOR_SOCKET', None)
            env['IMAGINE_PRIVACY_PYTHON'] = str(ROOT / '.venv-privacy/bin/python')
            env['HELTO_PRIVACY_SESSION_DIR'] = str(project / 'unused-helto-session')
            binary = str(ROOT / 'target/debug/imagine-server')
            os.execve(binary, [binary, '--project', str(project), '--bind', f'127.0.0.1:{self.port}', *args], env)
    def expect(self, text):
        until = time.monotonic() + 15
        while text.encode() not in self.output:
            if time.monotonic() > until: raise AssertionError('Expected synthetic server state timed out')
            if select.select([self.fd], [], [], .1)[0]:
                try: part = os.read(self.fd, 8192)
                except OSError: raise AssertionError('Synthetic server exited unexpectedly') from None
                if not part: raise AssertionError('Synthetic server exited unexpectedly')
                self.output += part
    def send(self, value):
        self.output = b''
        os.write(self.fd, value.encode() + b'\n')
    def request(self, path, body=None):
        access = json.loads((self.project / '.imagine/access.json').read_text())
        r = urllib.request.Request(f'http://127.0.0.1:{self.port}{path}', data=body,
            headers={'Authorization': 'Bearer ' + access['clients']['local'], 'Content-Type': 'image/png'})
        with urllib.request.urlopen(r, timeout=3) as response:
            return json.load(response)
    def stop(self):
        if self.pid is None: return
        try: os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        until = time.monotonic() + 10
        while time.monotonic() < until:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.pid = None
                os.close(self.fd)
                return status
            time.sleep(.05)
        os.kill(self.pid, signal.SIGKILL)
        os.waitpid(self.pid, 0)
        self.pid = None
        os.close(self.fd)
        raise AssertionError('Synthetic server did not stop')

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='imagine-privacy-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)
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
    def test_password_tty_gate_encryption_shutdown_and_wrong_password(self):
        self.stale()
        server = self.start()
        server.expect('New Imagine password:')
        self.assert_purged()
        with socket.socket() as s:
            self.assertNotEqual(s.connect_ex(('127.0.0.1', server.port)), 0)
        server.send('synthetic-only-password')
        server.expect('Repeat password:')
        server.send('synthetic-only-password')
        server.expect('Imagine listening')
        self.assertNotIn(b'synthetic-only-password', server.output)
        session = server.request('/v1/session')
        self.assertTrue(session['encrypted'])
        asset = server.request('/v1/assets', png())
        path = self.root / '.imagine/media' / session['id'] / asset['id']
        self.assertTrue(path.read_bytes().startswith(b'IMAGINE1'))
        self.assertEqual(server.stop(), 0)
        self.assertFalse(path.exists())
        self.assertTrue((self.root / '.imagine/privacy-keystore.json').exists())
        wrong = self.start()
        wrong.expect('Imagine password:')
        wrong.send('wrong-synthetic-password')
        wrong.expect('Imagine remains locked')
        with socket.socket() as s:
            self.assertNotEqual(s.connect_ex(('127.0.0.1', wrong.port)), 0)
    def test_plain_start_wipes_before_api_and_cleans_on_stop(self):
        self.stale()
        server = self.start('--no-encryption')
        server.expect('Imagine listening')
        self.assert_purged()
        session = server.request('/v1/session')
        self.assertFalse(session['encrypted'])
        asset = server.request('/v1/assets', png())
        path = self.root / '.imagine/media' / session['id'] / asset['id']
        self.assertEqual(path.read_bytes(), png())
        self.assertEqual(server.stop(), 0)
        self.assertFalse(path.exists())

if __name__ == '__main__': unittest.main()
