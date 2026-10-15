"""Print disk evidence and stop before a shared-machine evaluation exceeds its budget.

python e2e/disk_guard.py --owned PATH [--owned PATH ...] --windows-build PATH
Counts retained environments/builds as well as replay output; never deletes anything.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def size(root):
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                total += path.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned", type=Path, action="append", default=[])
    parser.add_argument("--windows-build", type=Path)
    args = parser.parse_args()
    subprocess.run(["df", "-h", "/", "/mnt/c"], check=True)
    for path in args.owned:
        if path.exists():
            subprocess.run(["du", "-sh", str(path)], check=True)
    sizes = {str(path): size(path) for path in args.owned}
    if args.windows_build:
        runtime = args.windows_build / "ito/runtime/python.exe"
        native = subprocess.check_output(
            ["wslpath", "-w", str(args.windows_build)], text=True
        ).strip()
        sizes[str(args.windows_build)] = int(
            subprocess.check_output(
                [
                    str(runtime),
                    "-I",
                    "-c",
                    "import os,sys;print(sum(os.path.getsize(os.path.join(d,n)) "
                    "for d,_,ns in os.walk(sys.argv[1]) for n in ns))",
                    native,
                ],
                text=True,
            )
        )
    free = shutil.disk_usage("/mnt/c").free
    print(
        json.dumps(dict(bytes=sizes, total_bytes=sum(sizes.values()), windows_free_bytes=free)),
        flush=True,
    )
    if free < 50_000_000_000 or sum(sizes.values()) > 19_500_000_000:
        raise SystemExit("Stop: C: free space or job budget would leave insufficient headroom")


if __name__ == "__main__":
    main()
