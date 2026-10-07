"""Run with: uv run python e2e/pairing.py. A LAN observer records the plain-HTTP signaling.

A proxy between a real pilot link and a real CLI driver keeps every /pairing and /offer
exchange, as a passive observer on the network would. The first pairing's six-digit code
falls to an offline search, but the driver already retired it. The reconnect that follows
uses no code, and its recorded proof matches none of the million codes.
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from ito.driver import pairing
from ito.link import PairingError, connect
from ito.link.pairing import proof
from ito.protocol import VERSION, Paired

ROOT = Path(__file__).resolve().parents[1]


class Observer:
    """Forwards signaling to the driver and keeps a copy of each exchange."""

    def __init__(self, target):
        self.target = target
        self.transcript = []

    async def forward(self, request):
        body = await request.read()
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                self.target + request.path,
                data=body,
                headers={"Content-Type": request.content_type},
            ) as answer,
        ):
            reply = await answer.read()
            self.transcript.append((request.path, body, answer.status, reply))
            return web.Response(status=answer.status, body=reply, content_type=answer.content_type)

    async def start(self):
        app = web.Application()
        app.router.add_post("/pairing", self.forward)
        app.router.add_post("/offer", self.forward)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    def last_pairing(self):
        """The nonce and offer of the newest accepted pairing."""
        offers = [i for i, entry in enumerate(self.transcript) if entry[0] == "/offer"]
        index = next(i for i in reversed(offers) if self.transcript[i][2] == 200)
        nonce = json.loads(self.transcript[index - 1][3])["nonce"]
        return nonce, json.loads(self.transcript[index][1])


def search(nonce, offer):
    """Every six-digit code whose proof matches the recorded offer."""
    return [
        f"{guess:06d}"
        for guess in range(1_000_000)
        if proof(f"{guess:06d}", nonce, "offer", offer["sdp"]) == offer["proof"]
    ]


async def attempt(address, sdp, key, pilot=None):
    """A fresh challenge answered with key's proof: what an observer holding key could send."""
    async with aiohttp.ClientSession() as session:
        async with session.post(address + "/pairing") as response:
            nonce = (await response.json())["nonce"]
        payload = {"version": VERSION, "type": "offer", "sdp": sdp, "nonce": nonce}
        payload |= {"proof": proof(key, nonce, "offer", sdp)} | ({"pilot": pilot} if pilot else {})
        async with session.post(address + "/offer", json=payload) as response:
            return response.status, await response.text()


async def pilot(address, **key):
    peer = await connect(address, **key)
    assert peer.connected and peer.description.name and peer.clock.offset is not None
    return peer


async def run():
    with tempfile.TemporaryDirectory(prefix="ito-e2e-pairing-") as directory:
        code_file = Path(directory) / "pairing-code"
        code = pairing.rotate(code_file)
        env = os.environ | {"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "e2e")}
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from ito.driver.cli import main; main()",
            "robot:Robot",
            "--adapter-args",
            json.dumps({"journal": str(Path(directory) / "robot.jsonl")}),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--pairing-file",
            str(code_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=ROOT,
            env=env,
        )
        try:
            async with asyncio.timeout(20):
                address = (await driver.stdout.readline()).decode().split()[-1]
                assert (await driver.stdout.readline()).decode().startswith("Pairing code: ")
            observer = Observer(address)
            relay = await observer.start()

            # First pairing: the pilot types the code and receives its own secret.
            peer = await pilot(relay, code=code)
            credential = peer.credential
            assert credential is not None and peer.send(Paired(pilot=credential.pilot))
            async with asyncio.timeout(5):
                while not pairing.pilots(code_file)["pilots"]:  # noqa: ASYNC110
                    await asyncio.sleep(0.02)
            assert pairing.pilots(code_file) == {
                "used": code,
                "pilots": {credential.pilot: credential.secret},
            }
            await peer.close()
            stored = (code_file.parent / "pairing-code.pilots").stat().st_mode
            assert os.name == "nt" or stored & 0o077 == 0

            # The observer recovers the code from that one exchange; it is already used up.
            nonce, offer = observer.last_pairing()
            assert "pilot" not in offer
            started = time.perf_counter()
            assert search(nonce, offer) == [code]
            first_search = time.perf_counter() - started
            status, reason = await attempt(address, offer["sdp"], code)
            assert status == 403 and "already used" in reason, (status, reason)
            try:
                await pilot(address, code=code)
                raise AssertionError("a used pairing code connected")
            except PairingError as exc:
                assert "already used" in str(exc), exc
            shown = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "from ito.driver.cli import main; main()",
                "--pairing-file",
                str(code_file),
                "--show-code",
                stdout=asyncio.subprocess.PIPE,
                cwd=ROOT,
            )
            shown = (await shown.communicate())[0].decode()
            assert "used by a paired pilot" in shown and code not in shown.replace(" ", ""), shown

            # Reconnect without the code: the pilot proves its secret through the observer.
            observer.transcript.clear()
            peer = await pilot(relay, credential=credential)
            assert not peer.credential_received.is_set()
            await peer.close()
            nonce, offer = observer.last_pairing()
            assert offer["pilot"] == credential.pilot
            assert search(nonce, offer) == [], "a code matches a secret-keyed proof"
            # The recorded offer cannot be replayed, and its identity alone proves nothing.
            async with (
                aiohttp.ClientSession() as session,
                session.post(address + "/offer", json=offer) as response,
            ):
                assert response.status == 400, response.status
            for key in (code, "0" * 32):
                status, reason = await attempt(address, offer["sdp"], key, offer["pilot"])
                assert (status, reason) == (403, "This robot no longer knows this pilot")

            # Rotating the code forgets every pilot; the new code pairs once more.
            rotated = pairing.rotate(code_file)
            try:
                await pilot(address, credential=credential)
                raise AssertionError("a forgotten pilot connected")
            except PairingError as exc:
                assert "no longer knows this pilot" in str(exc), exc
            peer = await pilot(address, code=rotated)
            assert peer.credential.secret != credential.secret
            await peer.close()
            print(
                json.dumps(
                    {
                        "event": "verified",
                        "first_pairing_code_recovered": True,
                        "recovered_code_refused": True,
                        "code_search_seconds": round(first_search, 2),
                        "reconnect_codes_matching": 0,
                        "rotation_forgets_pilots": True,
                    }
                )
            )
            await observer.runner.cleanup()
        finally:
            if driver.returncode is None:
                driver.send_signal(signal.SIGTERM)
                try:
                    async with asyncio.timeout(5):
                        await driver.wait()
                except TimeoutError:
                    driver.kill()
                    await driver.wait()
            assert driver.returncode == 0, f"driver shutdown failed: {driver.returncode}"


if __name__ == "__main__":
    asyncio.run(run())
