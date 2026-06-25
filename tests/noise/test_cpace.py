"""CPACE-X25519-SHA512 tests, including known-answer vectors from the draft.

The KAT values are the ``G_25519`` group object test vector from
draft-irtf-cfrg-cpace (``testvectors.json``), exercising generator derivation,
scalar multiplication, the shared secret ``K``, and the ISK transcript hash.
"""

from __future__ import annotations

import hashlib

import pytest

from aiosendspin.noise.cpace import (
    _DSI_ISK,
    CPace,
    CPaceError,
    CPaceRole,
    _calculate_generator,
    _lv_cat,
    _scalar_mult,
    _scalar_mult_vfy,
)


def _h(value: str) -> bytes:
    return bytes.fromhex(value)


# draft-irtf-cfrg-cpace testvectors.json -> "G_25519"
PRS = _h("50617373776F7264")
CI = _h("0B415F696E69746961746F720B425F726573706F6E646572")
SID = _h("7E4B4791D6A8EF019B936C79FB7F2C57")
GENERATOR = _h("D04BF6D41F6A289632A2E929FA29BEBD51092512A7829FDDE7D314B62F05A73F")
YA = _h("21B4F4BD9E64ED355C3EB676A28EBEDAF6D8F17BDC365995B319097153044080")
ADA = _h("414461")
YA_PUB = _h("1D13C89278CDADD826F6D8D7F887701430F8380DDC17611CDD6DC989CE0C9F32")
YB = _h("848B0779FF415F0AF4EA14DF9DD1D3C29AC41D836C7808896C4EBA19C51AC40A")
ADB = _h("414462")
YB_PUB = _h("248CCCF6D5CDC3646F0AD593F9E6CEF4E69D4945F8372E623512ECEA32185623")
K = _h("5B067EFFBDC0B2A0E1D907B21EBB25CFEDB96A852179A847C37E43EE71322C6B")
ISK_IR = _h(
    "6E19B875F7A561D6B3CA3DBB9EF42AC55DE3E717881018204B8922B4D5E53BB2"
    "AA82C300BEA7B65D2B671DA71922DDF6472301B79BC270ADFA8BF413285F2263"
)


def test_kat_calculate_generator() -> None:
    """Generator derivation matches the draft G_25519 vector."""
    assert _calculate_generator(PRS, CI, SID) == GENERATOR


def test_kat_public_shares() -> None:
    """Ya and Yb match X25519(scalar, generator) for the draft vector."""
    assert _scalar_mult(YA, GENERATOR) == YA_PUB
    assert _scalar_mult(YB, GENERATOR) == YB_PUB


def test_kat_shared_secret_both_sides() -> None:
    """Both sides derive the draft's shared secret K."""
    assert _scalar_mult_vfy(YA, YB_PUB) == K
    assert _scalar_mult_vfy(YB, YA_PUB) == K


def test_kat_isk_ir() -> None:
    """The initiator-responder ISK matches the draft vector."""
    transcript = _lv_cat(YA_PUB, ADA) + _lv_cat(YB_PUB, ADB)
    isk = hashlib.sha512(_lv_cat(_DSI_ISK, SID, K) + transcript).digest()
    assert isk == ISK_IR


# X25519_points "Invalid Y0" (all zero) — a low-order point.
LOW_ORDER_POINT = _h("0000000000000000000000000000000000000000000000000000000000000000")


def test_scalar_mult_vfy_rejects_low_order_point() -> None:
    """A low-order peer share aborts with CPaceError."""
    with pytest.raises(CPaceError):
        _scalar_mult_vfy(YA, LOW_ORDER_POINT)


def test_round_trip_matching_pin_confirms() -> None:
    """Matching PINs produce mutually verifying confirmation tags."""
    sid = b"sendspin-pair-pake-v1" + b"\x11" * 32
    pin = b"12345678"
    server = CPace.start(role=CPaceRole.INITIATOR, prs=pin, sid=sid)
    client = CPace.start(role=CPaceRole.RESPONDER, prs=pin, sid=sid)

    server.derive(client.public_share)
    client.derive(server.public_share)

    assert client.verify(server.tag())
    assert server.verify(client.tag())


def test_round_trip_mismatched_pin_fails_confirmation() -> None:
    """Mismatched PINs fail confirmation on both sides."""
    sid = b"sendspin-pair-pake-v1" + b"\x22" * 32
    server = CPace.start(role=CPaceRole.INITIATOR, prs=b"12345678", sid=sid)
    client = CPace.start(role=CPaceRole.RESPONDER, prs=b"87654321", sid=sid)

    server.derive(client.public_share)
    client.derive(server.public_share)

    assert not client.verify(server.tag())
    assert not server.verify(client.tag())


def test_tag_before_derive_raises() -> None:
    """Requesting a tag before derive raises CPaceError."""
    server = CPace.start(role=CPaceRole.INITIATOR, prs=b"12345678", sid=b"s")
    with pytest.raises(CPaceError):
        server.tag()


def test_derive_rejects_wrong_length_share() -> None:
    """A wrong-length peer share is rejected."""
    server = CPace.start(role=CPaceRole.INITIATOR, prs=b"12345678", sid=b"s")
    with pytest.raises(CPaceError):
        server.derive(b"too-short")
