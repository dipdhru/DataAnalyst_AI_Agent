"""The demo's URL vetting: every case here tries to make the server dial somewhere it shouldn't."""

from __future__ import annotations

import socket

import pytest

from analyst.netguard import UnsafeHostError, safe_postgres_url

PUBLIC = "93.184.216.34"


def resolving_to(*addresses: str):
    def resolver(host, port, type=0):  # same shape as socket.getaddrinfo
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    return resolver


def fails(raw: str, resolver=None) -> str:
    with pytest.raises(UnsafeHostError) as err:
        safe_postgres_url(raw, resolver=resolver or resolving_to(PUBLIC))
    return str(err.value)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.1.2.3",  # private ranges
        "172.16.0.9",
        "192.168.1.5",
        "169.254.169.254",  # cloud metadata service
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:10.0.0.1",  # private IPv4 inside an IPv6 address
        "::ffff:127.0.0.1",
    ],
)
def test_refuses_non_public_addresses(address: str) -> None:
    message = fails("postgresql://u:secret@db.example.com/app", resolving_to(address))
    assert "private or local" in message


def test_refuses_a_name_with_any_private_answer() -> None:
    # A public answer next to a private one would let the driver pick the private one.
    fails("postgresql://u:p@db.example.com/app", resolving_to(PUBLIC, "10.0.0.5"))


def test_refuses_ip_literals_that_are_private() -> None:
    fails("postgresql://u:p@192.168.0.10/app", resolving_to("192.168.0.10"))


def test_refuses_unresolvable_host() -> None:
    def broken(host, port, type=0):
        raise socket.gaierror("no such host")

    assert "Could not find" in fails("postgresql://u:p@nope.invalid/app", broken)


def test_pins_the_address_it_checked() -> None:
    url = safe_postgres_url(
        "postgresql://u:p@db.example.com:6543/app", resolver=resolving_to(PUBLIC)
    )
    assert url.query["hostaddr"] == PUBLIC  # connects to this IP, not a fresh DNS answer
    assert url.host == "db.example.com"  # still the name, so TLS can verify the certificate
    assert url.port == 6543
    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "require"
    assert url.query["connect_timeout"] == "10"


def test_accepts_the_postgres_scheme_alias_and_stricter_tls() -> None:
    url = safe_postgres_url(
        "postgres://u:p@db.example.com/app?sslmode=verify-full", resolver=resolving_to(PUBLIC)
    )
    assert url.query["sslmode"] == "verify-full"


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "bogus"])
def test_refuses_plaintext_tls_modes(mode: str) -> None:
    fails(f"postgresql://u:p@db.example.com/app?sslmode={mode}")


@pytest.mark.parametrize(
    "option",
    [
        "host=10.0.0.1",  # would override the host we checked
        "hostaddr=127.0.0.1",
        "options=-c%20search_path%3Dpg_catalog",
        "service=internal",
        "passfile=/etc/passwd",
        "sslrootcert=/etc/ssl/x",
    ],
)
def test_refuses_url_options_that_redirect_or_reach_files(option: str) -> None:
    fails(f"postgresql://u:p@db.example.com/app?{option}")


@pytest.mark.parametrize(
    "raw",
    [
        "sqlite:////etc/passwd",
        "mysql://u:p@db.example.com/app",
        "postgresql+psycopg2://u:p@db.example.com/app",
        "mongodb://u:p@db.example.com/app",
        "file:///etc/passwd",
        "not a url",
        "",
        "   ",
        "postgresql://u:p@/app",  # no host: libpq would use a local socket
        "postgresql://u:p@db1.example.com,10.0.0.1/app",  # multi-host list
        "postgresql://u:p@db.example.com",  # no database
        "postgresql://u:p@db.example.com/app?" + "a=" * 1500,
    ],
)
def test_refuses_other_schemes_and_malformed_urls(raw: str) -> None:
    fails(raw)


def test_errors_never_contain_the_password() -> None:
    for raw, resolver in [
        ("postgresql://u:hunter2@10.0.0.1/app", resolving_to("10.0.0.1")),
        ("postgresql://u:hunter2@db.example.com/app?sslmode=disable", resolving_to(PUBLIC)),
        ("postgresql://u:hunter2@db.example.com/app?host=x", resolving_to(PUBLIC)),
        ("postgresql://u:hunter2@db.example.com", resolving_to(PUBLIC)),
    ]:
        assert "hunter2" not in fails(raw, resolver)
