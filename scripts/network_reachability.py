#!/usr/bin/env python3
"""Tracks which domains are reachable on which networks, so a network-level
block (corporate firewall, DNS/SNI filtering) surfaces as a fast, clear
message instead of a slow timeout followed by a cryptic stack trace.

Built 2026-07-13 after huggingface.co was silently TLS-reset on Gil's work
network while building the Anthropic Fellows paper's reference stack: DNS
resolved fine, TCP connected fine, but the TLS handshake was reset right
after the Client Hello (SNI-level filtering) -- while api.semanticscholar.org
worked normally from the same machine at the same time. That failure mode
looks identical to "site is down" unless you specifically test at the TLS
layer, and it's specific to network, not machine -- the same MacBook Pro
works fine on other networks, and the Mac Mini reached huggingface.co fine
over its own (different) network at the same moment.

Network identity: gateway MAC address, not SSID. macOS redacts the SSID
from system_profiler/scutil output without a Location Services entitlement
most callers don't have; gateway MAC needs no special permission and,
unlike gateway IP, doesn't collide across the many routers that default to
the same private IP (192.168.1.1 etc.) -- it survives DHCP lease renewal
too, unlike relying on this machine's own IP.
"""
from __future__ import annotations
import json
import re
import socket
import ssl
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path.home() / ".claude" / "network-reachability.json"


def _default_gateway() -> str | None:
    out = subprocess.run(
        ["route", "-n", "get", "default"], capture_output=True, text=True
    ).stdout
    m = re.search(r"gateway:\s*(\S+)", out)
    return m.group(1) if m else None


def _arp_mac(ip: str) -> str | None:
    out = subprocess.run(["arp", "-n", ip], capture_output=True, text=True).stdout
    # macOS's arp does NOT zero-pad single-digit octets (e.g. "b4:c:25:e2:80:19",
    # not "b4:0c:..."), so a fixed 17-char match misses real output -- match
    # 6 variable-length hex groups instead.
    m = re.search(r"at ([0-9a-fA-F]{1,2}(?::[0-9a-fA-F]{1,2}){5})", out)
    return m.group(1) if m else None


def current_network_id() -> str:
    """Stable-enough identifier for 'which network am I on right now'.
    Falls back to the gateway IP if no arp entry exists yet (e.g. right
    after joining a network, before this machine has ARP'd the router),
    and to "unknown" if there's no default route at all (offline)."""
    gw = _default_gateway()
    if not gw:
        return "unknown"
    return _arp_mac(gw) or gw


def check_domain(domain: str, port: int = 443, timeout: float = 5.0) -> bool:
    """Real TLS-handshake-level reachability check. A plain DNS lookup or
    bare socket.connect() would both report success for the exact failure
    mode this module exists to catch (SNI-based filtering resets the
    connection during the TLS handshake itself, after DNS and TCP both
    succeed) -- so this goes one layer deeper and completes a real
    handshake, or doesn't."""
    try:
        with socket.create_connection((domain, port), timeout=timeout) as sock:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(sock, server_hostname=domain):
                return True
    except Exception:
        return False


def _load_store() -> dict:
    try:
        return json.loads(STORE_PATH.read_text())
    except Exception:
        return {}


def _save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2))


def record(domain: str, reachable: bool, network_id: str | None = None) -> None:
    network_id = network_id or current_network_id()
    store = _load_store()
    store.setdefault(network_id, {})[domain] = {
        "reachable": reachable,
        "last_checked": datetime.now(timezone.utc).isoformat(),
    }
    _save_store(store)


def known_status(domain: str, network_id: str | None = None) -> bool | None:
    """Prior recorded history for the current network, without a live
    check. None means this domain has never been checked on this network
    -- distinct from False (checked and found blocked), so a caller can
    tell "no data yet" from "known bad" and decide whether to check live."""
    network_id = network_id or current_network_id()
    entry = _load_store().get(network_id, {}).get(domain)
    return entry["reachable"] if entry else None


def check_and_record(domain: str, network_id: str | None = None) -> bool:
    network_id = network_id or current_network_id()
    reachable = check_domain(domain)
    record(domain, reachable, network_id)
    return reachable


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Check and remember domain reachability per network."
    )
    ap.add_argument("command", choices=["check", "status", "id"])
    ap.add_argument("domain", nargs="?")
    args = ap.parse_args()

    if args.command == "id":
        print(current_network_id())
        return

    if not args.domain:
        raise SystemExit("domain is required for 'check' and 'status'")

    if args.command == "check":
        reachable = check_and_record(args.domain)
        state = "reachable" if reachable else "BLOCKED"
        print(f"{args.domain}: {state} on network {current_network_id()}")
    elif args.command == "status":
        status = known_status(args.domain)
        if status is None:
            print(f"{args.domain}: no history on this network yet")
        else:
            print(f"{args.domain}: {'reachable' if status else 'BLOCKED'} (last recorded)")


if __name__ == "__main__":
    main()
