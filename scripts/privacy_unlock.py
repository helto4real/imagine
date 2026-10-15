#!/usr/bin/env python3
"""Interactive startup gate. Secrets never use argv, environment, stdout or session files.

Two independent AES-GCM slots wrap one random verifier: scrypt(password) and
Helto's verified FIDO2 hmac-secret provider. This is a dedicated Imagine store;
it never opens or modifies the user's shared Helto keystore/session.
"""
import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import warnings

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

AAD = b'imagine.startup-auth.v1'
N, R, P = 2**17, 8, 1


def b64(data):
    return base64.b64encode(data).decode('ascii')


def unb64(value):
    return base64.b64decode(value, validate=True)


def password_key(password, salt):
    if not 8 <= len(password) or len(password.encode()) > 4096:
        raise ValueError('Password must contain at least 8 characters and at most 4096 bytes.')
    return Scrypt(salt=salt, length=32, n=N, r=R, p=P).derive(password.encode())


def wrap(key, verifier, method):
    nonce = secrets.token_bytes(12)
    return {'nonce': b64(nonce), 'wrapped': b64(AESGCM(key).encrypt(nonce, verifier, AAD + method.encode()))}


def unwrap(key, slot, method):
    value = AESGCM(key).decrypt(unb64(slot['nonce']), unb64(slot['wrapped']), AAD + method.encode())
    if len(value) != 32:
        raise ValueError('Invalid verifier.')
    return value


def create(password):
    salt = secrets.token_bytes(16)
    slot = wrap(password_key(password, salt), secrets.token_bytes(32), 'password')
    slot['salt'] = b64(salt)
    return {'schema': 'imagine.startup-auth', 'version': 1, 'password': slot}


def unlock(store, method, credential, provider=None):
    if store.get('schema') != 'imagine.startup-auth' or store.get('version') != 1:
        raise ValueError('Unsupported Imagine keystore.')
    if method == 'password':
        slot = store['password']
        salt = unb64(slot['salt'])
        if len(salt) != 16:
            raise ValueError('Invalid salt.')
        key = password_key(credential, salt)
    elif method == 'yubikey':
        from helto_privacy.fido2_provider import Fido2Identity, YubiKeyFido2Provider
        slot = store['yubikey']
        identity = Fido2Identity(**{k: (v if k == 'public_key_sha256' else unb64(v)) for k, v in slot['identity'].items()})
        key = (provider or YubiKeyFido2Provider()).derive(identity, credential)
    else:
        raise ValueError('Unknown unlock method.')
    return unwrap(key, slot, method)


def enroll(store, verifier, pin, provider=None):
    from helto_privacy.fido2_provider import YubiKeyFido2Provider
    from dataclasses import asdict
    enrolled = (provider or YubiKeyFido2Provider()).enroll(pin=pin)
    slot = wrap(enrolled.secret, verifier, 'yubikey')
    slot['identity'] = {k: (v if isinstance(v, str) else b64(v)) for k, v in asdict(enrolled.identity).items()}
    # Prove the new slot before replacing anything. Password slot stays intact.
    if unwrap(enrolled.secret, slot, 'yubikey') != verifier:
        raise ValueError('YubiKey verification failed.')
    return {**store, 'yubikey': slot}


def save(path, value):
    fd, temp = tempfile.mkstemp(prefix='.privacy-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def secret(label):
    # No fallback to echoing stdin. Interactive secrets belong only to the TTY.
    with open('/dev/tty', 'w') as terminal:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            return getpass.getpass(label, stream=terminal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--method', choices=['password', 'yubikey'], default='password')
    parser.add_argument('--enroll-yubikey', action='store_true')
    args = parser.parse_args()
    path = args.store
    if path.is_symlink():
        raise ValueError('Refusing a symlink keystore.')
    if not path.exists():
        if args.method != 'password':
            raise ValueError('Set up the password first, then enroll your YubiKey.')
        password = secret('New Imagine password: ')
        if password != secret('Repeat password: '):
            raise ValueError('Passwords do not match.')
        store = create(password)
        save(path, store)
        verifier = unlock(store, 'password', password)
    else:
        with path.open('rb') as f:
            raw = f.read(65537)
        if len(raw) > 65536:
            raise ValueError('Invalid keystore.')
        store = json.loads(raw)
        if args.method not in store:
            raise ValueError('Unlock method is not enrolled.')
        label = 'Imagine password: ' if args.method == 'password' else 'YubiKey PIN (then touch the key): '
        verifier = unlock(store, args.method, secret(label))
    if args.enroll_yubikey:
        pin = secret('YubiKey PIN (enrollment requires two touches): ')
        save(path, enroll(store, verifier, pin))
        print('YubiKey enrolled; password unlock remains available.', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt):
        # Library exception messages can contain paths/device details. Never dump
        # inputs, a traceback, or key material. One attempt, no automatic PIN retry.
        print('Imagine unlock failed or cancelled. Check the password/PIN, enrolled key and privacy dependencies.', file=sys.stderr)
        sys.exit(1)
