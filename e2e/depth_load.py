"""Run with: uv run python e2e/depth_load.py. Status and clock sync keep flowing under depth.

A robot sends incompressible VGA depth with every video frame, about 25 MB/s of frame
metadata, more than aiortc's SCTP carries. Frame metadata only enters an idle association
and is dropped otherwise, so the 10 Hz Status must keep arriving (past 2 s without it the
pilot tears the link down) and the clock round trip must stay small: the driver judges
pilot input freshness through that clock.
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from ito import clock
from ito.driver import pairing
from ito.link import connect
from ito.protocol import FrameMetadata, Status

ROOT = Path(__file__).resolve().parents[1]
SECONDS = 10


async def run():
    with tempfile.TemporaryDirectory(prefix="ito-e2e-depth-") as directory:
        code_file = Path(directory) / "pairing-code"
        code = pairing.rotate(code_file)
        options = {"journal": str(Path(directory) / "robot.jsonl")}
        options |= {"resolution": [640, 480], "noisy_depth": True}
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from ito.driver.cli import main; main()",
            "robot:Robot",
            "--adapter-args",
            json.dumps(options),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--pairing-file",
            str(code_file),
            stdout=asyncio.subprocess.PIPE,
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "e2e")},
        )
        try:
            async with asyncio.timeout(20):
                address = (await driver.stdout.readline()).decode().split()[-1]
            async with await connect(address, code=code) as peer:
                statuses, frames, rtts = [], set(), []
                ended = clock.now() + SECONDS
                while (now := clock.now()) < ended:
                    try:
                        async with asyncio.timeout(ended - now):
                            message = await peer.messages.get()
                    except TimeoutError:
                        break
                    if isinstance(message, Status):
                        statuses.append(clock.now())
                        if peer.clock.rtt is not None:
                            rtts.append(peer.clock.rtt)
                    elif isinstance(message, FrameMetadata):
                        assert message.depth.width == 640
                        frames.add(message.sequence)
                assert peer.connected, "link lost under depth load"
            gaps = [b - a for a, b in zip(statuses, statuses[1:], strict=False)]
            result = {
                "statuses": len(statuses),
                "max_status_gap_ms": round(max(gaps) * 1000),
                "depth_frames": len(frames),
                "depth_mb_per_s": round(len(frames) * 640 * 480 * 2 * 4 / 3 / SECONDS / 1e6, 1),
                "max_clock_rtt_ms": round(max(rtts) * 1000, 1),
            }
            print(json.dumps(result))
            # Half the pilot's 2 s teardown; a shared reliable channel exceeded it here.
            assert len(statuses) >= SECONDS * 7, result
            assert max(gaps) < 1, result
            assert len(frames) >= SECONDS * 2, result
            # The driver refuses input older than 0.25 s by this clock; skew is half the round
            # trip. A depth backlog on the clock's path pushed it past 1 s.
            assert max(rtts) < 0.2, result
        finally:
            if driver.returncode is None:
                driver.send_signal(signal.SIGTERM)
                await driver.wait()


if __name__ == "__main__":
    asyncio.run(run())
