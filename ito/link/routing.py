"""Use the OS route to a direct robot, not an unrelated VPN interface."""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


async def direct_interface(pc, address: str) -> None:
    target = urlsplit(address)
    loop = asyncio.get_running_loop()
    resolved = await loop.getaddrinfo(target.hostname, target.port or 80, type=socket.SOCK_DGRAM)
    local = None
    for family, kind, protocol, _, endpoint in resolved:
        try:
            with socket.socket(family, kind, protocol) as sock:
                # UDP connect selects a route; it transmits no packet.
                sock.connect(endpoint)
                local = sock.getsockname()[0]
            break
        except OSError:
            continue
    if local is None or ipaddress.ip_address(local).is_loopback:
        return
    transports = {t.sender.transport.transport for t in pc.getTransceivers()}
    if pc.sctp:
        transports.add(pc.sctp.transport.transport)
    for transport in transports:
        # aiortc exposes no interface option. Keep this aioice compatibility seam
        # per connection: changing the global address enumerator races other peers.
        connection = transport.iceGatherer._connection
        gather = connection.get_component_candidates

        async def routed(component, addresses, timeout=5, gather=gather):
            if local not in addresses:
                raise ConnectionError("The network interface for this robot is unavailable")
            return await gather(component, [local], timeout)

        connection.get_component_candidates = routed
