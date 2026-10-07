"""Audit networking in the real pilot and spawned SLAM worker; allow the local robot."""

import multiprocessing
import os
import socket
import sys
from pathlib import Path

log = os.environ.get("ITO_E2E_NETWORK_LOG")
local = {"127.0.0.1", "::1", "localhost", socket.gethostname()}
local.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None))


def audit(event, args):
    worker = multiprocessing.current_process().name == "ito-reconstruction"
    attempted = event in ("urllib.Request", "http.client.connect")
    if event in ("socket.connect", "socket.sendto", "socket.getaddrinfo"):
        address = args[0] if event == "socket.getaddrinfo" else args[-1]
        host = address[0] if isinstance(address, tuple) else address
        attempted = worker or (
            isinstance(host, str) and host not in local and not host.startswith("/")
        )
    if attempted:
        with Path(log).open("a") as output:
            output.write(f"{os.getpid()} {event}\n")
        raise PermissionError("End-to-end pilot is offline except for the local robot")


if log:
    with Path(log).with_suffix(".pids").open("a") as output:
        output.write(f"{os.getpid()}\n")
    sys.addaudithook(audit)
