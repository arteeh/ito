"""uv run --with python-xlib --with pillow python e2e/microduck_viewer.py LAUNCHER [ARGS...]."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from Xlib import X, display, protocol


def windows(root):
    for window in root.query_tree().children:
        yield window
        yield from windows(window)


def main():
    out = Path("e2e/out/microduck-viewer")
    out.mkdir(parents=True, exist_ok=True)
    connection = display.Display()
    started = time.monotonic()
    with (out / "launcher.log").open("w") as log:
        child = subprocess.Popen(sys.argv[1:] + ["--viewer"], stdout=log, stderr=log)
        try:
            window = None
            while time.monotonic() - started < 30:
                assert child.poll() is None, "launcher exited; see launcher.log"
                window = next(
                    (
                        w
                        for w in windows(connection.screen().root)
                        if (w.get_wm_name() or "").startswith("MuJoCo :")
                        and w.get_attributes().map_state == X.IsViewable
                    ),
                    None,
                )
                if window:
                    break
                time.sleep(0.1)
            assert window, "Pollen's MuJoCo window did not appear"
            time.sleep(1)
            geometry = window.get_geometry()
            pixels = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xFFFFFFFF)
            assert pixels.depth in (24, 32), pixels.depth
            Image.frombytes(
                "RGB", (geometry.width, geometry.height), pixels.data, "raw", "BGRX"
            ).save(out / "viewer.png")
            window.send_event(
                protocol.event.ClientMessage(
                    window=window.id,
                    client_type=connection.intern_atom("WM_PROTOCOLS"),
                    data=(32, [connection.intern_atom("WM_DELETE_WINDOW"), X.CurrentTime, 0, 0, 0]),
                )
            )
            connection.flush()
            assert child.wait(timeout=5) == 0, "closing viewer did not stop the stack cleanly"
            report = {
                "viewer": "MuJoCo",
                "viewable": True,
                "closed": True,
                "elapsed_s": time.monotonic() - started,
            }
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            connection.close()


if __name__ == "__main__":
    main()
