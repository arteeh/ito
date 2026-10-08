"""Audit networking in the real pilot and spawned SLAM worker; allow the local robot."""

import multiprocessing
import os
import socket
import sys
from pathlib import Path

import psutil

log = os.environ.get("ITO_E2E_NETWORK_LOG")
# Every address this machine owns, whether or not its hostname resolves to it.
local = {"localhost", socket.gethostname()}
local.update(
    address.address.split("%")[0]
    for addresses in psutil.net_if_addrs().values()
    for address in addresses
    if address.family in (socket.AF_INET, socket.AF_INET6)
)


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
