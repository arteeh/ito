"""Pairing proofs: both ends show they share a key without sending it.

Each proof is an HMAC over a single-use driver nonce and the session description it
accompanies, so a captured proof cannot be replayed or moved to another offer, and the
answer's proof tells the pilot it reached the robot it paired with.

The six-digit code is the key once: anyone who records that exchange can search all codes
offline, so the driver retires the code as soon as the pilot holds a 128-bit secret of its
own, sent over the encrypted data channel. Every later proof uses that secret.
"""

import hashlib
import hmac
import re
import secrets

DIGITS = 6


class PairingError(ConnectionError):
    """The driver refused the pilot's pairing code or credential (HTTP 401 or 403)."""


def generate() -> str:
    return f"{secrets.randbelow(10**DIGITS):0{DIGITS}d}"


def token() -> str:
    """128 random bits: a pilot identity or secret."""
    return secrets.token_hex(16)


def normalize(code: str) -> str | None:
    """Accept what a pilot types ("482 913", "482-913"); None if it cannot be a code."""
    digits = re.sub(r"[\s-]", "", code)
    return digits if re.fullmatch(rf"[0-9]{{{DIGITS}}}", digits) else None


def display(code: str) -> str:
    return f"{code[:3]} {code[3:]}"


def proof(key: str, nonce: str, kind: str, sdp: str) -> str:
    message = "\n".join(("ito-pairing-1", kind, nonce, sdp)).encode()
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def valid(key: str, nonce: str, kind: str, sdp: str, value: str | None) -> bool:
    return value is not None and hmac.compare_digest(proof(key, nonce, kind, sdp), value)
