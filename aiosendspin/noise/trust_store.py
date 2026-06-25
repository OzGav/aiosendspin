"""Trust model and pairing-record stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from aiosendspin.models.types import PairMethod, TrustLevel
from aiosendspin.noise.keys import (
    PSK_SIZE,
    b64url_decode,
    b64url_encode,
    generate_psk,
    psk_id_for,
)
from aiosendspin.noise.pin import DEFAULT_MIN_PIN_DIGITS

# A PIN-pairing method enters terminal lockout when its failure counter reaches
# this value.
PIN_LOCKOUT_THRESHOLD: Final[int] = 10

__all__ = [
    "PIN_LOCKOUT_THRESHOLD",
    "ClientPairingConfig",
    "ClientPairingRecord",
    "ClientPairingStore",
    "InMemoryClientPairingStore",
    "InMemoryServerPairingStore",
    "PairingPsk",
    "PskCategory",
    "ResolvedPsk",
    "ServerPairingRecord",
    "ServerPairingStore",
    "StorageExhaustedError",
    "StorageReport",
    "TrustLevel",
]


class PskCategory(StrEnum):
    """Which kind of PSK was matched during a handshake."""

    LONG_TERM = "long_term"
    """A per-pair long-term PSK established through a successful pairing."""
    PAIRING = "pairing"
    """A Pairing PSK distributed out-of-band to admit a new client."""
    SENTINEL = "sentinel"
    """The published Sentinel PSK — used for PIN pairing and unpaired playback."""


class StorageExhaustedError(Exception):
    """A pairing cannot persist its record and has no shared-PSK fallback."""


@dataclass(frozen=True, slots=True)
class StorageReport:
    """A bounded client's record-storage accounting."""

    capacity: int
    free: int
    cost_individual: int
    cost_shared: int


@dataclass(frozen=True, slots=True)
class ResolvedPsk:
    """A PSK selected during a handshake, with its trust metadata."""

    psk_id: str
    psk: bytes
    category: PskCategory
    counterparty_id: str | None = None
    """Peer's ``client_id``/``server_id`` for stored-pubkey records; ``None`` otherwise."""


@dataclass(frozen=True, slots=True)
class ServerPairingRecord:
    """A long-term credential a server stores for one client."""

    psk_id: str
    psk: bytes
    client_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``long_term``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.LONG_TERM, self.client_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url, timestamp ISO-8601)."""
        return {
            "psk_id": self.psk_id,
            "psk": b64url_encode(self.psk),
            "client_id": self.client_id,
            "created_at": self.created_at.isoformat(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ServerPairingRecord:
        """Reconstruct a record from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            client_id=_str(data, "client_id"),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
            label=_opt_str(data, "label"),
        )


@dataclass(frozen=True, slots=True)
class ClientPairingRecord:
    """A long-term credential a client stores for a server."""

    psk_id: str
    psk: bytes
    server_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    label: str | None = None
    used: bool = False

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``long_term``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.LONG_TERM, self.server_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url, timestamp ISO-8601)."""
        return {
            "psk_id": self.psk_id,
            "psk": b64url_encode(self.psk),
            "server_id": self.server_id,
            "created_at": self.created_at.isoformat(),
            "label": self.label,
            "used": self.used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClientPairingRecord:
        """Reconstruct a record from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            server_id=_opt_str(data, "server_id"),
            created_at=datetime.fromisoformat(_str(data, "created_at")),
            label=_opt_str(data, "label"),
            used=_bool(data, "used"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClientPairingConfig:
    """Pairing policy a client persists."""

    pairing_psk_enabled: bool = True
    dynamic_pin_enabled: bool = True
    static_pin_enabled: bool = False
    unpaired_access_enabled: bool = False
    dynamic_pin_min_length: int = DEFAULT_MIN_PIN_DIGITS
    record_mode_psk_id: str
    """Shared-PSK record used as the storage-exhaustion fallback when pairing."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {
            "pairing_psk_enabled": self.pairing_psk_enabled,
            "dynamic_pin_enabled": self.dynamic_pin_enabled,
            "static_pin_enabled": self.static_pin_enabled,
            "unpaired_access_enabled": self.unpaired_access_enabled,
            "dynamic_pin_min_length": self.dynamic_pin_min_length,
            "record_mode_psk_id": self.record_mode_psk_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClientPairingConfig:
        """Reconstruct from ``to_dict`` output (defaults for absent keys)."""
        return cls(
            pairing_psk_enabled=_bool(data, "pairing_psk_enabled", default=True),
            dynamic_pin_enabled=_bool(data, "dynamic_pin_enabled", default=True),
            static_pin_enabled=_bool(data, "static_pin_enabled", default=False),
            unpaired_access_enabled=_bool(data, "unpaired_access_enabled", default=False),
            dynamic_pin_min_length=_int(
                data, "dynamic_pin_min_length", default=DEFAULT_MIN_PIN_DIGITS
            ),
            record_mode_psk_id=_str(data, "record_mode_psk_id"),
        )


@dataclass(frozen=True, slots=True)
class PairingPsk:
    """A Pairing PSK a client accepts to admit a server."""

    psk_id: str
    psk: bytes
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate the PSK size."""
        _check_psk(self.psk)

    def as_resolved(self) -> ResolvedPsk:
        """Project to the handshake-time currency (category ``pairing``)."""
        return ResolvedPsk(self.psk_id, self.psk, PskCategory.PAIRING, None)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (PSK base64url)."""
        return {"psk_id": self.psk_id, "psk": b64url_encode(self.psk), "label": self.label}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PairingPsk:
        """Reconstruct a Pairing PSK from ``to_dict`` output."""
        return cls(
            psk_id=_str(data, "psk_id"),
            psk=b64url_decode(_str(data, "psk")),
            label=_opt_str(data, "label"),
        )


class ServerPairingStore(ABC):
    """Long-term records and operator-staged Pairing PSKs, both keyed by ``client_id``."""

    @abstractmethod
    async def record_by_client_id(self, client_id: str) -> ServerPairingRecord | None:
        """Return the long-term record for ``client_id``, if any."""

    @abstractmethod
    async def store_record(self, record: ServerPairingRecord) -> None:
        """Persist a long-term record (keyed by its ``client_id``)."""

    @abstractmethod
    async def remove_record(self, client_id: str) -> None:
        """Remove the long-term record for ``client_id`` (no-op if absent)."""

    @abstractmethod
    async def staged_pairing_psk(self, client_id: str) -> PairingPsk | None:
        """Return the Pairing PSK staged to admit ``client_id`` for pairing, if any."""

    @abstractmethod
    async def stage_pairing_psk(self, client_id: str, pairing_psk: PairingPsk) -> None:
        """Stage an operator-entered Pairing PSK to admit ``client_id`` for pairing."""

    @abstractmethod
    async def unstage_pairing_psk(self, client_id: str) -> None:
        """Remove the staged Pairing PSK for ``client_id`` (no-op if absent)."""


class ClientPairingStore(ABC):
    """Pairing state a client holds: long-term records plus its accepted Pairing PSKs."""

    @abstractmethod
    async def resolve_by_psk_id(self, psk_id: str) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` to its PSK for the handshake, or ``None``."""

    @abstractmethod
    async def record_by_psk_id(self, psk_id: str) -> ClientPairingRecord | None:
        """Return the long-term record identified by ``psk_id``, if any."""

    @abstractmethod
    async def record_by_server_id(self, server_id: str) -> ClientPairingRecord | None:
        """Return the stored-pubkey record bound to ``server_id``, if any."""

    @abstractmethod
    async def store_record(self, record: ClientPairingRecord) -> None:
        """Persist a long-term record."""

    @abstractmethod
    async def remove_record(self, psk_id: str) -> None:
        """Remove the long-term record identified by ``psk_id`` (no-op if absent)."""

    @abstractmethod
    async def mark_record_used(self, psk_id: str) -> None:
        """Flag the record at ``psk_id`` as used (no-op if absent)."""

    @abstractmethod
    async def records(self) -> Sequence[ClientPairingRecord]:
        """Return all stored long-term records."""

    @abstractmethod
    async def get_pairing_config(self) -> ClientPairingConfig:
        """Return the persisted pairing policy (defaults if unset)."""

    @abstractmethod
    async def store_pairing_config(self, config: ClientPairingConfig) -> None:
        """Persist the pairing policy."""

    @abstractmethod
    async def set_pairing_psk(self, pairing_psk: PairingPsk) -> None:
        """Set the accepted Pairing PSK, replacing any existing one (admits a server with it)."""

    @abstractmethod
    async def clear_pairing_psk(self) -> None:
        """Remove the accepted Pairing PSK (no-op if absent)."""

    @abstractmethod
    async def pairing_psk(self) -> PairingPsk | None:
        """Return the accepted Pairing PSK, if any."""

    @abstractmethod
    async def set_static_pin(self, pin: str) -> None:
        """Set the configured static PIN (8 decimal digits), replacing any existing one."""

    @abstractmethod
    async def clear_static_pin(self) -> None:
        """Remove the configured static PIN (no-op if absent)."""

    @abstractmethod
    async def static_pin(self) -> str | None:
        """Return the configured static PIN, if any."""

    @abstractmethod
    async def pin_failure_count(self, method: PairMethod) -> int:
        """Return the persisted PIN-pairing failure count for ``method``."""

    @abstractmethod
    async def record_pin_failure(self, method: PairMethod) -> int:
        """Increment ``method``'s failure counter and return the new count."""

    @abstractmethod
    async def reset_pin_failures(self, method: PairMethod) -> None:
        """Reset ``method``'s failure counter to zero (on success or lockout clear)."""

    @abstractmethod
    async def is_pin_locked_out(self, method: PairMethod) -> bool:
        """Return whether ``method`` is in terminal lockout (count past the threshold)."""

    async def can_store_record(self) -> bool:
        """Return whether the store can persist another record (default: unlimited)."""
        return True

    async def storage_accounting(self) -> StorageReport | None:
        """Return record-storage accounting, or ``None`` if storage is unbounded/unknown."""
        return None

    async def resolve_pairing_outcome(
        self,
        *,
        server_id: str,
        label: str | None = None,
    ) -> tuple[bytes, ClientPairingRecord | None]:
        """Decide a pairing's outcome: a fresh per-server record, or the shared-PSK fallback."""
        if await self.can_store_record():
            psk = generate_psk()
            record = ClientPairingRecord(
                psk_id=psk_id_for(psk),
                psk=psk,
                server_id=server_id,
                label=label,
            )
            return psk, record
        # Storage exhausted: admit under the shared-PSK fallback record.
        psk_id = (await self.get_pairing_config()).record_mode_psk_id
        resolved = await self.resolve_by_psk_id(psk_id)
        if resolved is None or not _is_shared_record(resolved):
            msg = f"shared-PSK fallback record {psk_id!r} is missing or not shared"
            raise StorageExhaustedError(msg)
        return resolved.psk, None

    async def set_record_mode_psk_id(self, psk_id: str) -> None:
        """Set the shared-PSK fallback record; ``psk_id`` must name a shared record."""
        resolved = await self.resolve_by_psk_id(psk_id)
        if resolved is None:
            msg = f"record_mode psk_id {psk_id!r} references no record"
            raise ValueError(msg)
        if not _is_shared_record(resolved):
            msg = f"record_mode psk_id {psk_id!r} must reference a shared-PSK record"
            raise ValueError(msg)
        config = await self.get_pairing_config()
        await self.store_pairing_config(replace(config, record_mode_psk_id=psk_id))

    async def _record_mode_references(self, psk_id: str) -> bool:
        """Return whether the record_mode fallback references ``psk_id``."""
        return (await self.get_pairing_config()).record_mode_psk_id == psk_id

    async def can_remove_record(self, psk_id: str) -> bool:
        """Return whether the record at ``psk_id`` may be removed (not record_mode-referenced)."""
        return not await self._record_mode_references(psk_id)


class InMemoryServerPairingStore(ServerPairingStore):
    """Non-persistent ``ServerPairingStore`` (tests, ephemeral servers)."""

    def __init__(self) -> None:
        """Start with empty record and staged-Pairing-PSK tables."""
        self._records: dict[str, ServerPairingRecord] = {}
        self._staged: dict[str, PairingPsk] = {}

    async def record_by_client_id(self, client_id: str) -> ServerPairingRecord | None:
        """Return the long-term record for ``client_id``, if any."""
        return self._records.get(client_id)

    async def store_record(self, record: ServerPairingRecord) -> None:
        """Persist a long-term record keyed by its ``client_id``."""
        self._records[record.client_id] = record

    async def remove_record(self, client_id: str) -> None:
        """Remove the long-term record for ``client_id`` (no-op if absent)."""
        self._records.pop(client_id, None)

    async def staged_pairing_psk(self, client_id: str) -> PairingPsk | None:
        """Return the Pairing PSK staged to admit ``client_id``, if any."""
        return self._staged.get(client_id)

    async def stage_pairing_psk(self, client_id: str, pairing_psk: PairingPsk) -> None:
        """Stage an operator-entered Pairing PSK to admit ``client_id``."""
        self._staged[client_id] = pairing_psk

    async def unstage_pairing_psk(self, client_id: str) -> None:
        """Remove the staged Pairing PSK for ``client_id`` (no-op if absent)."""
        self._staged.pop(client_id, None)


class InMemoryClientPairingStore(ClientPairingStore):
    """Non-persistent ``ClientPairingStore`` (tests, ephemeral clients).

    Records are keyed by ``psk_id`` so the handshake-time lookup is O(1);
    ``record_by_server_id`` scans (the record count per client is small).
    """

    def __init__(self) -> None:
        """Start with a pre-provisioned shared-PSK fallback record and no other state."""
        self._records: dict[str, ClientPairingRecord] = {}
        self._pairing_psk: PairingPsk | None = None
        self._static_pin: str | None = None
        self._pin_failures: dict[PairMethod, int] = {}
        shared_psk = generate_psk()
        shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk)
        self._records[shared.psk_id] = shared
        self._pairing_config = ClientPairingConfig(record_mode_psk_id=shared.psk_id)

    async def resolve_by_psk_id(self, psk_id: str) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` (long-term record first, then the accepted Pairing PSK)."""
        record = self._records.get(psk_id)
        if record is not None:
            return record.as_resolved()
        if self._pairing_psk is not None and self._pairing_psk.psk_id == psk_id:
            return self._pairing_psk.as_resolved()
        return None

    async def record_by_psk_id(self, psk_id: str) -> ClientPairingRecord | None:
        """Return the long-term record identified by ``psk_id`` (O(1))."""
        return self._records.get(psk_id)

    async def record_by_server_id(self, server_id: str) -> ClientPairingRecord | None:
        """Return the stored-pubkey record bound to ``server_id`` (linear scan)."""
        for record in self._records.values():
            if record.server_id == server_id:
                return record
        return None

    async def store_record(self, record: ClientPairingRecord) -> None:
        """Persist a long-term record keyed by its ``psk_id``."""
        self._records[record.psk_id] = record

    async def remove_record(self, psk_id: str) -> None:
        """Remove the long-term record identified by ``psk_id`` (no-op if absent)."""
        self._records.pop(psk_id, None)

    async def mark_record_used(self, psk_id: str) -> None:
        """Flag the record at ``psk_id`` as used (no-op if absent)."""
        record = self._records.get(psk_id)
        if record is not None and not record.used:
            self._records[psk_id] = replace(record, used=True)

    async def records(self) -> Sequence[ClientPairingRecord]:
        """Return all stored long-term records."""
        return list(self._records.values())

    async def get_pairing_config(self) -> ClientPairingConfig:
        """Return the pairing policy."""
        return self._pairing_config

    async def store_pairing_config(self, config: ClientPairingConfig) -> None:
        """Persist the pairing policy."""
        self._pairing_config = config

    async def set_pairing_psk(self, pairing_psk: PairingPsk) -> None:
        """Set the accepted Pairing PSK, replacing any existing one."""
        self._pairing_psk = pairing_psk

    async def clear_pairing_psk(self) -> None:
        """Remove the accepted Pairing PSK (no-op if absent)."""
        self._pairing_psk = None

    async def pairing_psk(self) -> PairingPsk | None:
        """Return the accepted Pairing PSK, if any."""
        return self._pairing_psk

    async def set_static_pin(self, pin: str) -> None:
        """Set the configured static PIN, replacing any existing one."""
        self._static_pin = pin

    async def clear_static_pin(self) -> None:
        """Remove the configured static PIN (no-op if absent)."""
        self._static_pin = None

    async def static_pin(self) -> str | None:
        """Return the configured static PIN, if any."""
        return self._static_pin

    async def pin_failure_count(self, method: PairMethod) -> int:
        """Return the PIN-pairing failure count for ``method``."""
        return self._pin_failures.get(method, 0)

    async def record_pin_failure(self, method: PairMethod) -> int:
        """Increment ``method``'s failure counter and return the new count."""
        count = self._pin_failures.get(method, 0) + 1
        self._pin_failures[method] = count
        return count

    async def reset_pin_failures(self, method: PairMethod) -> None:
        """Reset ``method``'s failure counter to zero."""
        self._pin_failures.pop(method, None)

    async def is_pin_locked_out(self, method: PairMethod) -> bool:
        """Return whether ``method`` has reached terminal lockout."""
        return self._pin_failures.get(method, 0) >= PIN_LOCKOUT_THRESHOLD


# --- private helpers -----------------------------------------------------


def _is_shared_record(resolved: ResolvedPsk) -> bool:
    """Return whether ``resolved`` is a shared-PSK record (long-term, no counterparty)."""
    return resolved.category is PskCategory.LONG_TERM and resolved.counterparty_id is None


def _check_psk(psk: bytes) -> None:
    if len(psk) != PSK_SIZE:
        msg = f"PSK must be {PSK_SIZE} bytes, got {len(psk)}"
        raise ValueError(msg)


def _bool(data: Mapping[str, object], key: str, *, default: bool | None = None) -> bool:
    value = data[key] if default is None else data.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key!r} must be a boolean, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _int(data: Mapping[str, object], key: str, *, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key!r} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        msg = f"{key!r} must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _opt_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key!r} must be a string or null, got {type(value).__name__}"
        raise TypeError(msg)
    return value
