"""The driver's pairing code: generated once, kept in a private file, checked on every offer."""

import os
import secrets
import time
from collections import deque
from pathlib import Path

from ito.link.pairing import generate, normalize, valid

NONCE_SECONDS = 30
MAX_NONCES = 32
MAX_FAILURES = 10  # Wrong codes per minute before every offer is refused for that minute.


class PairingRefused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status


def default_path() -> Path:
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
    else:
        root = os.environ.get("XDG_STATE_HOME")
    return (Path(root) if root else Path.home() / ".local/state") / "ito" / "pairing-code"


def write(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        f.write(code + "\n")
    temporary.replace(path)


def read(path: Path) -> str:
    code = normalize(path.read_text())
    if code is None:
        raise ValueError(f"{path} does not hold a pairing code; rotate it")
    return code


def ensure(path: Path) -> tuple[str, bool]:
    """The persisted code, and whether it was generated just now."""
    try:
        return read(path), False
    except FileNotFoundError:
        code = generate()
        write(path, code)
        return code, True


def rotate(path: Path) -> str:
    code = generate()
    write(path, code)
    return code


class Pairing:
    """Hands out single-use nonces and checks offer proofs against the code on disk.

    Reading the file per offer means a rotated code applies to the next connection
    without restarting the driver.
    """

    def __init__(self, path: Path):
        self.path = path
        self._nonces: dict[str, float] = {}
        self._failures: deque[float] = deque()

    def nonce(self) -> str:
        now = time.monotonic()
        self._nonces = {n: t for n, t in self._nonces.items() if now - t < NONCE_SECONDS}
        while len(self._nonces) >= MAX_NONCES:
            del self._nonces[next(iter(self._nonces))]
        nonce = secrets.token_hex(16)
        self._nonces[nonce] = now
        return nonce

    def check(self, nonce: str | None, proof: str | None, sdp: str) -> str:
        """The code the offer proved; raises PairingRefused with a reason the pilot reads."""
        now = time.monotonic()
        while self._failures and now - self._failures[0] > 60:
            self._failures.popleft()
        if len(self._failures) >= MAX_FAILURES:
            raise PairingRefused(429, "Too many wrong pairing codes; wait a minute and try again")
        if nonce is None or proof is None:
            raise PairingRefused(401, "This robot needs its pairing code")
        issued = self._nonces.pop(nonce, None)
        if issued is None or now - issued >= NONCE_SECONDS:
            raise PairingRefused(400, "Pairing challenge expired; try again")
        try:
            code = read(self.path)
        except (OSError, ValueError) as exc:
            raise PairingRefused(503, f"The robot cannot read its pairing code: {exc}") from None
        if not valid(code, nonce, "offer", sdp, proof):
            self._failures.append(now)
            raise PairingRefused(403, "Wrong pairing code")
        return code
