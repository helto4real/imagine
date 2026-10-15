"""Synthetic credentials and fake hardware only. Never accesses user keystores."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import privacy_unlock as privacy
from helto_privacy.fido2_provider import Fido2Identity, Fido2Enrollment

class Hardware:
    identity = Fido2Identity(b'credential', b'aaguid', b'public', 'fingerprint', b's' * 32)
    secret = b'y' * 32
    def enroll(self, *, pin):
        assert pin == 'synthetic-pin'
        return Fido2Enrollment(self.identity, self.secret)
    def derive(self, identity, pin):
        assert identity == self.identity
        if pin != 'synthetic-pin':
            raise ValueError('Incorrect fake PIN')
        return self.secret

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = privacy.create('synthetic-password')
    def test_password_and_yubikey_unlock_independently(self):
        key = privacy.unlock(self.store, 'password', 'synthetic-password')
        store = privacy.enroll(self.store, key, 'synthetic-pin', Hardware())
        self.assertEqual(store['password'], self.store['password'])
        self.assertEqual(privacy.unlock(store, 'password', 'synthetic-password'), key)
        self.assertEqual(privacy.unlock(store, 'yubikey', 'synthetic-pin', Hardware()), key)
        with self.assertRaises(Exception): privacy.unlock(store, 'password', 'wrong-password')
        with self.assertRaises(Exception): privacy.unlock(store, 'yubikey', 'wrong-pin', Hardware())
        other = Hardware(); other.secret = b'z' * 32
        with self.assertRaises(Exception): privacy.unlock(store, 'yubikey', 'synthetic-pin', other)
    def test_tampered_slot_rejected(self):
        store = copy.deepcopy(self.store)
        store['password']['wrapped'] = privacy.b64(b'x' * 48)
        with self.assertRaises(Exception): privacy.unlock(store, 'password', 'synthetic-password')
    def test_private_atomic_store_and_no_plain_password(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'synthetic-keystore.json'
            privacy.save(path, self.store)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('synthetic-password', path.read_text())
            self.assertEqual(len(list(Path(d).iterdir())), 1)
    def test_no_tty_never_falls_back_to_stdin(self):
        with patch('builtins.open', side_effect=OSError('no terminal')):
            with self.assertRaises(OSError): privacy.secret('Synthetic: ')

if __name__ == '__main__': unittest.main()
