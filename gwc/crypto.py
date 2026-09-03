"""Encryption of the ground-truth targets (Fernet: AES-128-CBC + HMAC-SHA256).

The *ciphertext* is committed to the (public) repository; the key never is.
The key lives in the local secrets directory (``$GWC_SECRETS_DIR``, default
``~/.gwc``). On the evaluation machine the grader reads it to decrypt targets
in memory; a submission is told not to read it (honesty model -- see README).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict

import numpy as np
from cryptography.fernet import Fernet


def secrets_dir() -> Path:
    return Path(os.environ.get("GWC_SECRETS_DIR", Path.home() / ".gwc")).expanduser()


def key_path() -> Path:
    return secrets_dir() / "gwc-fernet.key"


def load_or_create_key(path: Path | None = None) -> bytes:
    p = Path(path) if path else key_path()
    if p.exists():
        return p.read_bytes().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    p.write_bytes(key)
    os.chmod(p, 0o600)
    return key


def load_key(path: Path | None = None) -> bytes:
    p = Path(path) if path else key_path()
    if not p.exists():
        raise FileNotFoundError(
            f"no decryption key at {p}. The grader needs the key to score against the "
            "encrypted targets; it is not part of the repository."
        )
    return p.read_bytes().strip()


def encrypt_targets(targets: Dict[int, Dict[str, np.ndarray]], key: bytes) -> bytes:
    """Serialise {index: {'means','vars','n_samples'}} to an npz and encrypt it."""
    buf = io.BytesIO()
    arrays = {}
    for i, t in targets.items():
        arrays[f"m{i}"] = np.asarray(t["means"], dtype=np.float32)
        arrays[f"v{i}"] = np.asarray(t["vars"], dtype=np.float32)
        arrays[f"n{i}"] = np.asarray(int(t["n_samples"]), dtype=np.int64)
    np.savez_compressed(buf, **arrays)
    return Fernet(key).encrypt(buf.getvalue())


def decrypt_targets(blob: bytes, key: bytes) -> Dict[int, Dict[str, np.ndarray]]:
    raw = Fernet(key).decrypt(blob)
    z = np.load(io.BytesIO(raw))
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for k in z.files:
        if k[0] != "m":
            continue
        i = int(k[1:])
        out[i] = {"means": z[f"m{i}"], "vars": z[f"v{i}"], "n_samples": int(z[f"n{i}"])}
    return out
