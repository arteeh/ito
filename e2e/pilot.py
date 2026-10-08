import argparse
import asyncio
import base64
import json
import random
import zlib

from aiortc import AudioStreamTrack
from aiortc.mediastreams import MediaStreamError

from ito import clock
from ito.link import connect
from ito.protocol import VERSION, Command, Paired, PilotState, Pose, Status


async def run(address: str, code: str):
    async with await connect(address, audio=AudioStreamTrack(), code=code) as peer:
        assert peer.control.ordered and peer.control.maxRetransmits is None
        assert not peer.pilot.ordered and peer.pilot.maxRetransmits == 0
        channel = peer.frame_channel
        assert not channel.ordered and channel.maxPacketLifeTime and channel.maxRetransmits is None
        assert peer.send(Paired(pilot=peer.credential.pilot))
        frames = {"video": 0, "audio": 0}
        luma = set()
        sequence = 0
        command_sequence = 0

        async def media(track):
            try:
                while True:
                    frame = await track.recv()
                    frames[track.kind] += 1
                    if track.kind == "video":
                        luma.add(bytes(frame.planes[0])[0])
            except MediaStreamError:
                return

        async def tracks():
            async with asyncio.TaskGroup() as group:
                while True:
                    group.create_task(media(await peer.tracks.get()))

        media_task = asyncio.create_task(tracks())

        def state(deadman=True, capture=None):
            nonlocal sequence
            sequence += 1
            return PilotState(
                sequence=sequence,
                capture_time=clock.now() if capture is None else capture,
                deadman=deadman,
                head=Pose(position=(0.1, 0.2, 0.3)),
                hands={"left": Pose()},
                trackers={"waist": Pose()},
                buttons={"grip": True},
                axes={"forward": 0.7},
            )

        async def drive(duration=0.5, deadman=True):
            deadline = clock.now() + duration
            while clock.now() < deadline:
                peer.send(state(deadman))
                await asyncio.sleep(0.01)

        async def status(predicate):
            async with asyncio.timeout(5):
                while True:
                    message = await peer.messages.get()
                    if isinstance(message, Status) and predicate(message):
                        return message

        async def command(action):
            nonlocal command_sequence
            command_sequence += 1
            assert peer.send(Command(sequence=command_sequence, action=action))
            return await status(lambda s: s.command_sequence == command_sequence)

        try:
            # Deadman input moves nothing until the pilot resumes the new connection.
            await drive(0.3)
            await status(lambda s: s.state == "stopped")
            assert (await command("resume")).state == "neutral"
            await drive()
            active = await status(lambda s: s.state == "active")
            assert active.telemetry["applied"] > 0
            await asyncio.sleep(0.3)
            async with asyncio.timeout(5):
                while min(frames.values()) < 5 or len(luma) < 3 or not peer.frames:  # noqa: ASYNC110
                    await asyncio.sleep(0.02)
            metadata = peer.frames["front"]
            assert metadata.camera_pose.position == (1.0, 2.0, 3.0)
            assert metadata.depth.to_bytes() == b"\xe8\x03" * (160 * 120)
            assert peer.description.cameras[0].track_id == "front-video"
            assert peer.clock.rtt is not None and peer.clock.rtt < 0.2
            assert abs(clock.now() - peer.clock.remote_to_local(metadata.capture_time)) < 2

            released = state(False)
            peer.send(released)
            await status(lambda s: s.state == "neutral" and s.reason == "deadman released")
            assert (await command("e-stop")).state == "e-stopped"
            await drive(0.3)
            latched = await status(lambda s: s.state == "e-stopped")
            before = latched.telemetry["applied"]
            await drive(0.3)
            latched = await status(lambda s: s.state == "e-stopped")
            assert latched.telemetry["applied"] == before
            assert (await command("resume")).state == "neutral"
            await drive(0.3)
            await status(lambda s: s.state == "active")
            assert (await command("stop")).state == "stopped"
            await drive(0.2)
            await status(lambda s: s.state == "stopped")
            await command("resume")

            # Hundreds of mutations go through real SCTP into the running driver.
            baseline = (await status(lambda s: True)).rejected_messages
            rng = random.Random(1)
            valid = state().model_dump()
            payloads = [
                b"\xff\x00",
                "{",
                "[]",
                "null",
                "{}",
                '"hello"',
                json.dumps({k: v for k, v in valid.items() if k != "version"}),
            ]
            changes = {
                "version": [0, 1, VERSION + 1, True, "2", None],
                "type": ["robot", "unknown", 42],
                "deadman": [1, "true", None],
                "sequence": [-1, 1.5, "4"],
                "capture_time": [-1, "NaN", None, float("inf")],
                "head": [
                    {"position": [0, 0], "orientation": [0, 0, 0, 0]},
                    {"position": [0, 0, 0], "orientation": [0, 0, 0, 2]},
                ],
                "axes": [{"forward": 2}, {"forward": float("nan")}],
            }
            for _ in range(250):
                field = rng.choice(list(changes))
                mutated = valid | {field: rng.choice(changes[field])}
                payloads.append(json.dumps(mutated))
            # Valid schemas on the wrong channel must not become commands or input.
            payloads.append(Command(sequence=1000, action="resume").model_dump_json())
            for payload in payloads:
                peer.pilot.send(payload)
                await asyncio.sleep(0.002)
            peer.control.send(state().model_dump_json())
            peer.control.send(
                json.dumps(
                    {"version": VERSION + 1, "type": "command", "sequence": 500, "action": "resume"}
                )
            )
            peer.control.send("x" * 1_500_001)
            # A decompression bomb is rejected by the real driver decoder before direction checks.
            peer.control.send(
                json.dumps(
                    {
                        "version": VERSION,
                        "type": "frame",
                        "camera": "front",
                        "sequence": 1,
                        "capture_time": clock.now(),
                        "depth": {
                            "width": 1,
                            "height": 1,
                            "encoding": "zlib-u16-mm",
                            "data": base64.b64encode(zlib.compress(b"x" * 1_000_000)).decode(),
                        },
                    }
                )
            )
            rejected = await status(lambda s: s.rejected_messages >= baseline + len(payloads) + 4)
            assert rejected.state != "fault"

            # New sequence with old capture time must neither actuate nor refresh deadman.
            await drive(0.3)
            await status(lambda s: s.state == "active")
            peer.send(state(capture=clock.now() - 10))
            await status(lambda s: s.state == "neutral" and s.reason == "input timeout")
            await drive(0.4)
            await status(lambda s: s.state == "active")
            print(
                json.dumps(
                    {
                        "event": "verified",
                        "frames": frames,
                        "changing_video": len(luma),
                        "rejected": rejected.rejected_messages,
                        "rtt": peer.clock.rtt,
                        "offset": peer.clock.offset,
                        "credential": peer.credential.model_dump(include={"pilot", "secret"}),
                    }
                ),
                flush=True,
            )
            while True:
                await drive(0.5)
        finally:
            media_task.cancel()
            try:
                await media_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("code", help="the driver's pairing code")
    args = parser.parse_args()
    asyncio.run(run(args.address, args.code))
