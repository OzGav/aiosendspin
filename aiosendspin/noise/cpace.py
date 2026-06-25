"""CPACE-X25519-SHA512 PAKE with explicit mutual confirmation.

The ``CPACE-X25519-SHA512`` cipher suite of draft-irtf-cfrg-cpace in
initiator-responder mode: the server is role ``A``, the client is role ``B``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Final

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

# Curve25519 field and Elligator2 parameters (draft G_X25519).
_Q: Final[int] = 2**255 - 19
_A: Final[int] = 486662
_Z: Final[int] = 2  # the non-square used by Elligator2 on Curve25519
_FIELD_BYTES: Final[int] = 32
_SHARE_SIZE: Final[int] = 32

_DSI: Final[bytes] = b"CPace255"
_DSI_ISK: Final[bytes] = b"CPace255_ISK"
_MAC_LABEL: Final[bytes] = b"CPaceMac"
_SHA512_BLOCK_BYTES: Final[int] = 128

_INV2: Final[int] = pow(2, -1, _Q)
_LEGENDRE_POWER: Final[int] = (_Q - 1) // 2


class CPaceError(Exception):
    """A CPace step failed (bad peer share, or a confirmation-tag mismatch)."""


class CPaceRole(Enum):
    """CPace protocol role. The Sendspin server is ``A``, the client is ``B``."""

    INITIATOR = "A"
    RESPONDER = "B"


def _prepend_len(data: bytes) -> bytes:
    length = len(data)
    out = bytearray()
    while True:
        if length < 128:
            out.append(length)
        else:
            out.append((length & 0x7F) | 0x80)
        length >>= 7
        if length == 0:
            break
    return bytes(out) + data


def _lv_cat(*parts: bytes) -> bytes:
    return b"".join(_prepend_len(p) for p in parts)


def _generator_string(prs: bytes, ci: bytes, sid: bytes) -> bytes:
    len_zpad = max(
        0,
        _SHA512_BLOCK_BYTES - 1 - len(_prepend_len(prs)) - len(_prepend_len(_DSI)),
    )
    return _lv_cat(_DSI, prs, b"\x00" * len_zpad, ci, sid)


def _decode_u(value: bytes) -> int:
    u = bytearray(value)
    u[-1] &= 0x7F  # 255-bit field: ignore the unused top bit (RFC 7748)
    return int.from_bytes(u, "little")


def _elligator2(r: int) -> bytes:
    r %= _Q
    v = (-_A * pow((1 + _Z * r * r) % _Q, -1, _Q)) % _Q
    eps = pow((v * v * v + _A * v * v + v) % _Q, _LEGENDRE_POWER, _Q)  # B = 1
    x = (eps * v - (1 - eps) * _A * _INV2) % _Q
    return x.to_bytes(_FIELD_BYTES, "little")


def _calculate_generator(prs: bytes, ci: bytes, sid: bytes) -> bytes:
    gen_hash = hashlib.sha512(_generator_string(prs, ci, sid)).digest()[:_FIELD_BYTES]
    return _elligator2(_decode_u(gen_hash))


def _scalar_mult(scalar: bytes, point: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(scalar).exchange(
        X25519PublicKey.from_public_bytes(point),
    )


def _scalar_mult_vfy(scalar: bytes, point: bytes) -> bytes:
    """X25519 scalar mult that rejects a result encoding the identity (low order)."""
    try:
        shared = _scalar_mult(scalar, point)
    except ValueError as exc:  # cryptography rejects low-order points outright
        raise CPaceError("peer share encodes a low-order point") from exc
    if shared == bytes(_FIELD_BYTES):
        raise CPaceError("peer share encodes a low-order point")
    return shared


@dataclass(slots=True)
class CPace:
    """One side of a CPACE-X25519-SHA512 exchange with mutual confirmation."""

    _role: CPaceRole
    _scalar: bytes
    _sid: bytes
    _ad: bytes
    public_share: bytes
    """This side's CPace public share (``Ya`` for the initiator, ``Yb`` otherwise)."""
    _peer_ad: bytes
    _mac_key: bytes | None = None
    _initiator_share: bytes | None = None
    _initiator_ad: bytes | None = None
    _responder_share: bytes | None = None
    _responder_ad: bytes | None = None

    @classmethod
    def start(
        cls,
        *,
        role: CPaceRole,
        prs: bytes,
        sid: bytes,
        ci: bytes = b"",
        ad: bytes = b"",
        peer_ad: bytes = b"",
    ) -> CPace:
        """Begin a CPace run, sampling a scalar and computing the public share.

        ``prs`` is the password-related string (the PIN's ASCII digits for
        Sendspin), ``sid`` the session id, ``ci`` the channel identifier
        (empty for Sendspin), and ``ad``/``peer_ad`` this side's and the peer's
        associated data (both empty for Sendspin).
        """
        scalar = secrets.token_bytes(_FIELD_BYTES)
        share = _scalar_mult(scalar, _calculate_generator(prs, ci, sid))
        return cls(
            _role=role,
            _scalar=scalar,
            _sid=sid,
            _ad=ad,
            public_share=share,
            _peer_ad=peer_ad,
        )

    def derive(self, peer_share: bytes) -> None:
        """Ingest the peer's public share, deriving the confirmation MAC key."""
        if len(peer_share) != _SHARE_SIZE:
            raise CPaceError(f"peer share must be {_SHARE_SIZE} bytes, got {len(peer_share)}")
        shared = _scalar_mult_vfy(self._scalar, peer_share)
        if self._role is CPaceRole.INITIATOR:
            self._initiator_share, self._initiator_ad = self.public_share, self._ad
            self._responder_share, self._responder_ad = peer_share, self._peer_ad
        else:
            self._initiator_share, self._initiator_ad = peer_share, self._peer_ad
            self._responder_share, self._responder_ad = self.public_share, self._ad
        transcript = _lv_cat(self._initiator_share, self._initiator_ad) + _lv_cat(
            self._responder_share, self._responder_ad
        )
        isk = hashlib.sha512(_lv_cat(_DSI_ISK, self._sid, shared) + transcript).digest()
        self._mac_key = hashlib.sha512(_MAC_LABEL + self._sid + isk).digest()

    def tag(self) -> bytes:
        """Return this side's confirmation tag (``Ta`` for ``A``, ``Tb`` for ``B``)."""
        return self._mac(own=True)

    def verify(self, peer_tag: bytes) -> bool:
        """Return whether ``peer_tag`` matches the peer's expected confirmation tag."""
        return hmac.compare_digest(peer_tag, self._mac(own=False))

    def _mac(self, *, own: bool) -> bytes:
        if self._mac_key is None or self._initiator_share is None or self._responder_share is None:
            raise CPaceError("derive() must be called before computing confirmation tags")
        assert self._initiator_ad is not None
        assert self._responder_ad is not None
        # Ta authenticates (Ya, ADa); Tb authenticates (Yb, ADb).
        if own == (self._role is CPaceRole.INITIATOR):
            share, ad = self._initiator_share, self._initiator_ad
        else:
            share, ad = self._responder_share, self._responder_ad
        return hmac.new(self._mac_key, _lv_cat(share, ad), hashlib.sha512).digest()
