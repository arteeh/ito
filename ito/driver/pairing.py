"""The driver's pairing code and the pilots it let in, kept in private files.

The code file holds the six-digit code. Next to it, `<code file>.pilots` holds each paired
pilot's secret and the code they used: a code that paired a pilot is never accepted again,
so a recorded first pairing gives nothing reusable. --rotate-code arms a new code and
forgets every pilot.
"""

import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ito import clock
from ito.link.pairing import generate, normalize, token, valid

NONCE_SECONDS = 30
MAX_NONCES = 32
MAX_PILOTS = 16
# Wrong codes per minute from one address, then from everyone, before codes are refused.
MAX_FAILURES = 10
MAX_FAILURES_TOTAL = 100


class PairingRefused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status


@dataclass(frozen=True)
class Grant:
    key: str  # Proves the answer: the code or the pilot's secret.
    code: str | None  # The code this offer used; the pilot gets a credential for it.


def default_path() -> Path:
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
    else:
        root = os.environ.get("XDG_STATE_HOME")
    return (Path(root) if root else Path.home() / ".local/state") / "ito" / "pairing-code"


def pilots_path(path: Path) -> Path:
    return path.with_name(path.name + ".pilots")


def write(path: Path, text: str) -> None:
    """Atomic and private: mkstemp creates the file readable by its owner only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=".pairing-")
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path: Path) -> str:
    code = normalize(path.read_text())
    if code is None:
        raise ValueError(f"{path} does not hold a pairing code; rotate it")
    return code


def pilots(path: Path) -> dict:
    """{"used": code that already paired a pilot or None, "pilots": {pilot: secret}}."""
    try:
        state = json.loads(pilots_path(path).read_text())
    except FileNotFoundError:
        return {"used": None, "pilots": {}}
    if not (
        isinstance(state, dict)
        and (state.get("used") is None or isinstance(state["used"], str))
        and isinstance(state.get("pilots"), dict)
        and all(isinstance(v, str) for v in state["pilots"].values())
    ):
        raise ValueError(f"{pilots_path(path)} is damaged; rotate the pairing code")
    return state


def armed(path: Path) -> bool:
    """The code can still pair a pilot."""
    return pilots(path)["used"] != read(path)


def ensure(path: Path) -> tuple[str, bool]:
    """The persisted code, and whether it was generated just now."""
    try:
        return read(path), False
    except FileNotFoundError:
        code = generate()
        write(path, code + "\n")
        return code, True


def rotate(path: Path) -> str:
    try:
        previous = read(path)
    except (FileNotFoundError, ValueError):
        previous = None
    code = generate()
    while code == previous:
        code = generate()
    write(path, code + "\n")
    pilots_path(path).unlink(missing_ok=True)
    return code


class Pairing:
    """Hands out single-use nonces and checks offer proofs against the files on disk.

    Reading the files per offer means a rotated code applies to the next connection
    without restarting the driver.
    """

    def __init__(self, path: Path):
        self.path = path
        self._nonces: dict[str, float] = {}
        self._failures: dict[str, deque[float]] = {}
        self._all_failures: deque[float] = deque()

    def nonce(self) -> str:
        now = clock.now()
        self._nonces = {n: t for n, t in self._nonces.items() if now - t < NONCE_SECONDS}
        while len(self._nonces) >= MAX_NONCES:
            del self._nonces[next(iter(self._nonces))]
        nonce = token()
        self._nonces[nonce] = now
        return nonce

    def check(
        self, nonce: str | None, proof: str | None, sdp: str, pilot: str | None, remote: str
    ) -> Grant:
        """What the offer proved; raises PairingRefused with a reason the pilot reads."""
        now = clock.now()
        if nonce is None or proof is None:
            raise PairingRefused(401, "This robot needs its pairing code")
        issued = self._nonces.pop(nonce, None)
        if issued is None or now - issued >= NONCE_SECONDS:
            raise PairingRefused(400, "Pairing challenge expired; try again")
        try:
            state = pilots(self.path)
            code = None if pilot else read(self.path)
        except (OSError, ValueError) as exc:
            raise PairingRefused(503, f"The robot cannot read its pairing state: {exc}") from None
        if pilot:
            # A 128-bit secret cannot be guessed, so these offers never count as failures.
            secret = state["pilots"].get(pilot)
            if secret is None or not valid(secret, nonce, "offer", sdp, proof):
                raise PairingRefused(403, "This robot no longer knows this pilot")
            return Grant(secret, None)
        if state["used"] == code:
            raise PairingRefused(
                403, "This pairing code was already used; the robot's --rotate-code makes a new one"
            )
        self._limit(remote, now)
        if not valid(code, nonce, "offer", sdp, proof):
            self._failures.setdefault(remote, deque()).append(now)
            self._all_failures.append(now)
            raise PairingRefused(403, "Wrong pairing code")
        return Grant(code, code)

    def _limit(self, remote: str, now: float) -> None:
        """Count wrong codes per TCP peer address, so one host cannot lock the pilot out."""
        for failures in (*self._failures.values(), self._all_failures):
            while failures and now - failures[0] > 60:
                failures.popleft()
        self._failures = {address: f for address, f in self._failures.items() if f}
        if len(self._failures.get(remote, ())) >= MAX_FAILURES:
            raise PairingRefused(429, "Too many wrong pairing codes; wait a minute and try again")
        if len(self._all_failures) >= MAX_FAILURES_TOTAL:
            raise PairingRefused(
                429, "Too many wrong pairing codes on this network; wait a minute and try again"
            )

    def paired(self, code: str, pilot: str, secret: str) -> None:
        """Remember the pilot's secret and retire the code it paired with."""
        state = pilots(self.path)
        if read(self.path) != code:
            return  # Rotated while this pilot was connecting: it must enter the new code.
        remembered = {p: s for p, s in state["pilots"].items() if p != pilot}
        while len(remembered) >= MAX_PILOTS:
            del remembered[next(iter(remembered))]
        remembered[pilot] = secret
        write(pilots_path(self.path), json.dumps({"used": code, "pilots": remembered}) + "\n")
