"""Vet a Postgres URL typed by a stranger before the server connects to it.

In the public demo, visitors give us a connection string and our server dials it. Left unchecked
that is a way to reach hosts only the server can reach (localhost, the cloud metadata service,
the private network). So: only public addresses, only Postgres over TLS, and no URL options that
could override the host. The connection is pinned to the address we checked (`hostaddr`), so a
DNS answer that changes between the check and the connect can't redirect it.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

Resolver = Callable[..., list[Any]]

SSL_MODES = ("require", "verify-ca", "verify-full")  # never send credentials in the clear
CONNECT_TIMEOUT_S = "10"
MAX_URL_LENGTH = 2000


class UnsafeHostError(ValueError):
    """The URL is malformed or points somewhere the demo server must not connect to.
    Messages never contain the URL, so a password can't leak into the UI or logs."""


def safe_postgres_url(raw: str, *, resolver: Resolver = socket.getaddrinfo) -> URL:
    """Return a SQLAlchemy URL that is safe to connect to, or raise UnsafeHostError."""
    raw = (raw or "").strip()
    if not raw:
        raise UnsafeHostError("Enter a connection string.")
    if len(raw) > MAX_URL_LENGTH:
        raise UnsafeHostError("That connection string is too long.")
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    try:
        url = make_url(raw)
    except ArgumentError:
        raise UnsafeHostError(
            "Not a valid connection string. Expected postgresql://user:password@host:5432/dbname"
        ) from None
    if url.get_backend_name() != "postgresql" or url.get_driver_name() not in ("psycopg", None):
        raise UnsafeHostError("Only postgresql:// connection strings are supported.")
    if not url.database:
        raise UnsafeHostError("The connection string needs a database name after the host.")

    host = url.host or ""
    if not host or "," in host or host.startswith("/"):
        raise UnsafeHostError("Use a single host name or IP address.")
    port = url.port or 5432
    if not 1 <= port <= 65535:
        raise UnsafeHostError("The port must be between 1 and 65535.")

    sslmode = "require"
    for key, value in url.query.items():
        # Anything else (host=, hostaddr=, options=, service=, passfile=...) can redirect the
        # connection or run settings on the server, so only the TLS mode is accepted.
        if key != "sslmode":
            raise UnsafeHostError(f"The URL option {key!r} is not supported.")
        if value not in SSL_MODES:
            raise UnsafeHostError(f"sslmode must be one of: {', '.join(SSL_MODES)}.")
        sslmode = value

    address = _public_address(host, port, resolver)
    return url.set(
        drivername="postgresql+psycopg",
        port=port,
        query={"sslmode": sslmode, "hostaddr": address, "connect_timeout": CONNECT_TIMEOUT_S},
    )


def _public_address(host: str, port: int, resolver: Resolver) -> str:
    try:
        infos = resolver(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise UnsafeHostError("Could not find that host. Check the spelling.") from None
    addresses: list[str] = []
    for info in infos:
        text = str(info[4][0]).split("%")[0]  # drop an IPv6 zone id
        if text not in addresses:
            addresses.append(text)
    if not addresses:
        raise UnsafeHostError("Could not find that host. Check the spelling.")
    # Every answer must be public: a name that also resolves to a private address is refused.
    for text in addresses:
        if not _is_public(text):
            raise UnsafeHostError(
                "That host is on a private or local network. The demo server can only connect "
                "to databases reachable from the public internet."
            )
    return addresses[0]


def _is_public(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # ::ffff:10.0.0.1 is the private address in disguise
    return ip.is_global
