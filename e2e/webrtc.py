"""Run with: uv run python e2e/webrtc.py. Real CLI driver, pilot process and WebRTC."""

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import aiohttp

from ito import clock
from ito.driver import pairing
from ito.link.pairing import proof
from ito.protocol import VERSION

ROOT = Path(__file__).resolve().parents[1]


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def line(process, deadline=20):
    async with asyncio.timeout(deadline):
        value = await process.stdout.readline()
    if not value:
        await process.wait()
        raise RuntimeError(f"process exited {process.returncode}; see stderr")
    return value.decode().strip()


async def challenge(session, address):
    async with session.post(address + "/pairing") as response:
        assert response.status == 200
        return (await response.json())["nonce"]


async def guesses(address, codes, host="127.0.0.1"):
    """Offer each code from one local address; the driver sees that TCP peer address."""
    sdp = "v=0\r\n"
    connector = aiohttp.TCPConnector(local_addr=(host, 0))
    async with aiohttp.ClientSession(connector=connector) as session:
        replies = []
        for code in codes:
            nonce = await challenge(session, address)
            offer = {"version": VERSION, "type": "offer", "sdp": sdp, "nonce": nonce}
            offer["proof"] = proof(code, nonce, "offer", sdp)
            async with session.post(address + "/offer", json=offer) as response:
                replies.append((response.status, await response.text()))
        return replies


async def run():
    with tempfile.TemporaryDirectory(prefix="ito-e2e-") as directory:
        journal = Path(directory) / "robot.jsonl"
        code_file = Path(directory) / "pairing-code"
        code = pairing.rotate(code_file)
        env = os.environ | {"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "e2e")}
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from ito.driver.cli import main; main()",
            "robot:Robot",
            "--adapter-args",
            json.dumps({"journal": str(journal)}),
            "--port",
            "0",
            "--command-rate",
            "20",
            "--input-timeout",
            "0.25",
            "--pairing-file",
            str(code_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=ROOT,
            env=env,
        )
        pilot = None
        try:
            address = (await line(driver)).rsplit(" ", 1)[1]
            async with aiohttp.ClientSession() as session:
                for payload in [
                    {},
                    {"version": VERSION + 1, "type": "offer", "sdp": "invalid"},
                    {"version": True, "type": "offer", "sdp": "invalid"},
                ]:
                    async with session.post(address + "/offer", json=payload) as response:
                        assert response.status == 400
                async with session.post(address + "/offer", data="x" * 230_000) as response:
                    assert response.status == 413
                sdp = "v=0\r\n"
                offer = {"version": VERSION, "type": "offer", "sdp": sdp}
                async with session.post(address + "/offer", json=offer) as response:
                    assert response.status == 401
                    assert await response.text() == "This robot needs its pairing code"
                wrong = "111111" if code != "111111" else "222222"
                nonce = await challenge(session, address)
                paired = offer | {"nonce": nonce, "proof": proof(wrong, nonce, "offer", sdp)}
                async with session.post(address + "/offer", json=paired) as response:
                    assert response.status == 403
                    assert await response.text() == "Wrong pairing code"
                # A nonce answers one offer only, so a captured proof cannot be replayed.
                paired["proof"] = proof(code, nonce, "offer", sdp)
                async with session.post(address + "/offer", json=paired) as response:
                    assert response.status == 400, response.status
            # Wrong codes from one host lock out that host, not the pilot elsewhere on the LAN.
            replies = await guesses(address, [wrong] * 10 + [code], host="127.0.0.2")
            assert [status for status, _ in replies] == [403] * 10 + [429], replies
            pilot = await asyncio.create_subprocess_exec(
                sys.executable,
                str(ROOT / "e2e" / "pilot.py"),
                address,
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                cwd=ROOT,
                env=env,
            )
            result = json.loads(await line(pilot, 35))
            assert result["event"] == "verified"
            credential = result.pop("credential")
            async with aiohttp.ClientSession() as session:
                nonce = await challenge(session, address)
                busy = offer | {
                    "nonce": nonce,
                    "proof": proof(credential["secret"], nonce, "offer", sdp),
                    "pilot": credential["pilot"],
                }
                async with session.post(address + "/offer", json=busy) as response:
                    assert response.status == 409
            await asyncio.sleep(0.15)
            killed_at = clock.now()
            pilot.kill()
            await pilot.wait()
            async with asyncio.timeout(2):
                while True:
                    neutral = [
                        e
                        for e in events(journal)
                        if e["event"] == "neutral" and e["time"] >= killed_at
                    ]
                    if neutral:
                        break
                    await asyncio.sleep(0.005)
            latency = neutral[0]["time"] - killed_at
            assert latency <= 0.25, f"neutral took {latency:.3f}s"
            await asyncio.sleep(0.35)
            entries = events(journal)
            applied = [e for e in entries if e["event"] == "apply"]
            assert len(applied) >= 10
            assert all(
                b["time"] - a["time"] >= 0.049 for a, b in zip(applied, applied[1:], strict=False)
            )
            assert all(e["head"]["position"] == [0.1, 0.2, 0.3] for e in applied)
            assert not any(e["time"] > neutral[0]["time"] for e in applied)
            assert any(e["event"] == "incoming_audio" for e in entries)
            assert driver.returncode is None
            # The pilot's code paired it once; nobody can use it again, so guessing is moot.
            replies = await guesses(address, [code, wrong])
            assert all(status == 403 and "already used" in text for status, text in replies)
            # With a fresh code armed, many hosts together hit the network-wide ceiling.
            code = pairing.rotate(code_file)
            hosts = 0
            while True:
                hosts += 1
                replies = await guesses(address, [wrong] * 10, host=f"127.0.0.{2 + hosts}")
                if replies[-1][0] == 429:
                    break
                assert [status for status, _ in replies] == [403] * 10, replies
                assert hosts <= 10, "no network-wide limit on wrong codes"
            assert "on this network" in replies[-1][1], replies
            replies = await guesses(address, [code], host="127.0.0.200")
            assert replies[0][0] == 429, replies
            print(
                json.dumps(
                    result
                    | {
                        "neutral_after_kill_ms": round(latency * 1000, 1),
                        "applied_commands": len(applied),
                        "command_rate_hz": 20,
                        "hosts_to_network_limit": hosts,
                    }
                )
            )
        finally:
            if pilot and pilot.returncode is None:
                pilot.kill()
                await pilot.wait()
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
