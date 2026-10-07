"""Pairing proofs: both ends show they know the robot's code without sending it.

Each proof is an HMAC keyed by the code over a single-use driver nonce and the session
description it accompanies, so a captured proof cannot be replayed or moved to another
offer, and the answer's proof tells the pilot it reached the robot it paired with.
"""

import hashlib
import hmac
import re
import secrets

DIGITS = 6


class PairingError(ConnectionError):
    """The driver refused the pilot's pairing code, or could not prove it knows it."""


def generate() -> str:
    return f"{secrets.randbelow(10**DIGITS):0{DIGITS}d}"


def normalize(code: str) -> str | None:
    """Accept what a pilot types ("482 913", "482-913"); None if it cannot be a code."""
    digits = re.sub(r"[\s-]", "", code)
    return digits if re.fullmatch(rf"[0-9]{{{DIGITS}}}", digits) else None


def display(code: str) -> str:
    return f"{code[:3]} {code[3:]}"


def proof(code: str, nonce: str, kind: str, sdp: str) -> str:
    message = "\n".join(("ito-pairing-1", kind, nonce, sdp)).encode()
    return hmac.new(code.encode(), message, hashlib.sha256).hexdigest()


def valid(code: str, nonce: str, kind: str, sdp: str, value: str | None) -> bool:
    return value is not None and hmac.compare_digest(proof(code, nonce, kind, sdp), value)
