"""Tests for :mod:`aiosendspin.noise.trust_store`."""

from __future__ import annotations

import pytest

from aiosendspin.models.types import PairMethod
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    PIN_LOCKOUT_THRESHOLD,
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
)


def _server_record(client_id: str = "client-A", label: str | None = None) -> ServerPairingRecord:
    psk = generate_psk()
    return ServerPairingRecord(psk_id=psk_id_for(psk), psk=psk, client_id=client_id, label=label)


def _client_record(
    server_id: str = "server-X",
    label: str | None = None,
) -> ClientPairingRecord:
    psk = generate_psk()
    return ClientPairingRecord(
        psk_id=psk_id_for(psk),
        psk=psk,
        server_id=server_id,
        label=label,
    )


def _shared_record(
    label: str | None = None,
) -> ClientPairingRecord:
    psk = generate_psk()
    return ClientPairingRecord(
        psk_id=psk_id_for(psk),
        psk=psk,
        server_id=None,
        label=label,
    )


def _pairing_psk(label: str | None = None) -> PairingPsk:
    psk = generate_psk()
    return PairingPsk(psk_id=psk_id_for(psk), psk=psk, label=label)


class _ExhaustedClientStore(InMemoryClientPairingStore):
    """A client store with no capacity for new records (to exercise the fallback)."""

    async def can_store_record(self) -> bool:
        return False


def test_records_reject_wrong_psk_size() -> None:
    """Each record type enforces the 32-byte PSK invariant."""
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        ServerPairingRecord(psk_id="x", psk=b"short", client_id="c")
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        ClientPairingRecord(psk_id="x", psk=b"short", server_id="s")
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        PairingPsk(psk_id="x", psk=b"short")


def test_server_record_round_trips_and_resolves() -> None:
    """ServerPairingRecord to/from dict round-trips; as_resolved names the client."""
    record = _server_record(client_id="client-A", label="Phone")
    assert ServerPairingRecord.from_dict(record.to_dict()) == record
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id == "client-A"
    assert "trust" not in record.to_dict()  # server never persists trust level


def test_client_record_round_trips_and_resolves() -> None:
    """ClientPairingRecord to/from dict round-trips; as_resolved names the server."""
    record = _client_record(server_id="server-X", label="Hub")
    restored = ClientPairingRecord.from_dict(record.to_dict())
    assert restored == record
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id == "server-X"


def test_pairing_psk_round_trips_and_resolves() -> None:
    """PairingPsk to/from dict round-trips; as_resolved has no counterparty."""
    pairing = _pairing_psk(label="QR token")
    assert PairingPsk.from_dict(pairing.to_dict()) == pairing
    resolved = pairing.as_resolved()
    assert resolved.category is PskCategory.PAIRING
    assert resolved.counterparty_id is None


def test_psk_serialized_as_unpadded_base64url() -> None:
    """to_dict emits the PSK as an unpadded base64url string."""
    record = _server_record()
    psk_field = record.to_dict()["psk"]
    assert isinstance(psk_field, str)
    assert "=" not in psk_field


async def test_server_store_record_round_trip() -> None:
    """store_record then record_by_client_id returns the record by client_id."""
    store = InMemoryServerPairingStore()
    record = _server_record(client_id="client-A")
    await store.store_record(record)
    assert await store.record_by_client_id("client-A") == record
    assert await store.record_by_client_id("client-B") is None


async def test_server_store_remove_record() -> None:
    """remove_record deletes the record by client_id; removing an absent one is a no-op."""
    store = InMemoryServerPairingStore()
    await store.store_record(_server_record(client_id="client-A"))
    await store.remove_record("client-A")
    assert await store.record_by_client_id("client-A") is None
    await store.remove_record("client-A")  # no-op


async def test_client_pairing_psk_lifecycle() -> None:
    """The client's accepted Pairing PSK: set, look up, replace, clear."""
    store = InMemoryClientPairingStore()
    pairing = _pairing_psk()
    await store.set_pairing_psk(pairing)
    assert await store.pairing_psk() == pairing
    assert await store.resolve_by_psk_id(pairing.psk_id) == pairing.as_resolved()
    # Setting a new one replaces the old.
    other = _pairing_psk()
    await store.set_pairing_psk(other)
    assert await store.pairing_psk() == other
    assert await store.resolve_by_psk_id(pairing.psk_id) is None
    await store.clear_pairing_psk()
    assert await store.pairing_psk() is None
    assert await store.resolve_by_psk_id(other.psk_id) is None
    # Clearing when absent is a no-op.
    await store.clear_pairing_psk()


async def test_client_static_pin_lifecycle() -> None:
    """The client's configured static PIN: set, look up, replace, clear."""
    store = InMemoryClientPairingStore()
    assert await store.static_pin() is None
    await store.set_static_pin("12345678")
    assert await store.static_pin() == "12345678"
    await store.set_static_pin("87654321")
    assert await store.static_pin() == "87654321"
    await store.clear_static_pin()
    assert await store.static_pin() is None
    # Clearing when absent is a no-op.
    await store.clear_static_pin()


async def test_client_store_resolves_by_psk_id_and_finds_by_server_id() -> None:
    """ClientPairingStore resolves a record by psk_id and finds it by server_id."""
    store = InMemoryClientPairingStore()
    record = _client_record(server_id="server-X")
    await store.store_record(record)
    assert await store.resolve_by_psk_id(record.psk_id) == record.as_resolved()
    assert await store.record_by_server_id("server-X") == record
    assert await store.resolve_by_psk_id("nope") is None
    assert await store.record_by_server_id("server-Y") is None


async def test_client_store_record_takes_precedence_over_pairing_psk() -> None:
    """When a long-term record and a Pairing PSK share a psk_id, the record wins."""
    store = InMemoryClientPairingStore()
    record = _client_record(server_id="server-X")
    pairing = PairingPsk(psk_id=record.psk_id, psk=record.psk, label="dup")
    await store.set_pairing_psk(pairing)
    await store.store_record(record)
    assert await store.resolve_by_psk_id(record.psk_id) == record.as_resolved()


async def test_client_store_remove_and_list() -> None:
    """Removing deletes a record; records() reflects current contents."""
    store = InMemoryClientPairingStore()
    a = _client_record(server_id="server-A")
    b = _client_record(server_id="server-B")
    await store.store_record(a)
    await store.store_record(b)
    added = {r for r in await store.records() if r.server_id is not None}
    assert added == {a, b}
    await store.remove_record(a.psk_id)
    assert await store.resolve_by_psk_id(a.psk_id) is None
    added = {r for r in await store.records() if r.server_id is not None}
    assert added == {b}
    # Removing an absent record is a no-op.
    await store.remove_record("absent")


async def test_client_store_reports_no_storage_accounting_by_default() -> None:
    """The default store is unbounded and reports no storage accounting."""
    assert await InMemoryClientPairingStore().storage_accounting() is None


async def test_pin_failure_counter_increments_and_resets() -> None:
    """Failures accumulate per method and reset clears the counter."""
    store = InMemoryClientPairingStore()
    assert await store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 0
    assert await store.record_pin_failure(PairMethod.DYNAMIC_PIN) == 1
    assert await store.record_pin_failure(PairMethod.DYNAMIC_PIN) == 2
    await store.reset_pin_failures(PairMethod.DYNAMIC_PIN)
    assert await store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 0


async def test_pin_failure_counter_is_per_method() -> None:
    """static_pin and dynamic_pin counters are tracked independently."""
    store = InMemoryClientPairingStore()
    await store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    assert await store.pin_failure_count(PairMethod.STATIC_PIN) == 0
    assert await store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 1


async def test_pin_lockout_at_threshold_and_clears_on_reset() -> None:
    """Lockout trips at the threshold and clears only on reset."""
    store = InMemoryClientPairingStore()
    for _ in range(PIN_LOCKOUT_THRESHOLD - 1):
        await store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    assert not await store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)
    await store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    assert await store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)
    await store.reset_pin_failures(PairMethod.DYNAMIC_PIN)
    assert not await store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)


# --- shared-PSK records --------------------------------------------------


def test_shared_record_round_trips_and_resolves() -> None:
    """A shared-PSK record omits server_id; as_resolved carries no counterparty."""
    record = _shared_record(label="Living Room")
    assert record.server_id is None
    restored = ClientPairingRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.server_id is None
    resolved = record.as_resolved()
    assert resolved.category is PskCategory.LONG_TERM
    assert resolved.counterparty_id is None


async def test_shared_record_excluded_from_server_id_lookup() -> None:
    """A shared-PSK record is found by psk_id but never by server_id."""
    store = InMemoryClientPairingStore()
    record = _shared_record()
    await store.store_record(record)
    assert await store.resolve_by_psk_id(record.psk_id) == record.as_resolved()
    assert await store.record_by_server_id("any-server") is None


async def test_set_record_mode_psk_id_validates_reference() -> None:
    """The record_mode psk_id must reference an existing shared-PSK record."""
    store = InMemoryClientPairingStore()
    shared = _shared_record()
    pubkey = _client_record(server_id="server-X")
    await store.store_record(shared)
    await store.store_record(pubkey)

    # The store is pre-provisioned with a shared fallback at construction.
    initial = (await store.get_pairing_config()).record_mode_psk_id
    assert initial is not None
    assert await store.record_by_psk_id(initial) is not None

    with pytest.raises(ValueError, match="references no record"):
        await store.set_record_mode_psk_id("missing")
    with pytest.raises(ValueError, match="must reference a shared-PSK record"):
        await store.set_record_mode_psk_id(pubkey.psk_id)

    await store.set_record_mode_psk_id(shared.psk_id)
    assert (await store.get_pairing_config()).record_mode_psk_id == shared.psk_id
    assert not await store.can_remove_record(shared.psk_id)


async def test_resolve_outcome_mints_stored_pubkey_record() -> None:
    """A storable store generates a fresh PSK bound to the server_id."""
    store = InMemoryClientPairingStore()
    psk, record = await store.resolve_pairing_outcome(
        server_id="server-X",
        label="Hub",
    )
    assert record is not None
    assert record.psk == psk
    assert record.psk_id == psk_id_for(psk)
    assert record.server_id == "server-X"


async def test_resolve_outcome_falls_back_to_shared_on_exhaustion() -> None:
    """A configured shared fallback admits under the shared record when full."""
    store = _ExhaustedClientStore()
    shared = _shared_record()
    await store.store_record(shared)
    await store.set_record_mode_psk_id(shared.psk_id)
    psk, record = await store.resolve_pairing_outcome(
        server_id="server-X",
        label=None,
    )
    assert psk == shared.psk
    assert record is None
