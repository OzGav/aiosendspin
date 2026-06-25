"""Tests for :mod:`aiosendspin.noise.pairing`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.models.core import ServerActivateMessage, ServerActivatePayload
from aiosendspin.models.types import Activity, PairAbortReason, PairMethod
from aiosendspin.noise.keys import Identity, b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.models import (
    ClientPairAuthMessage,
    ClientPairAuthPayload,
    ClientPairInitMessage,
    ClientPairInitPayload,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
)
from aiosendspin.noise.pairing import (
    PairingAbortError,
    PairingError,
    run_dynamic_pin_client,
    run_dynamic_pin_server,
    run_pairing_psk_client,
    run_pairing_psk_server,
    run_static_pin_client,
    run_static_pin_server,
)
from aiosendspin.noise.session import NoiseCipherSuite, NoiseSession
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
)
from aiosendspin.noise.wire import EncryptedWebSocket


def _added_records(records: Sequence[ClientPairingRecord]) -> list[ClientPairingRecord]:
    """Stored-pubkey records added by pairing (excludes the pre-provisioned shared record)."""
    return [r for r in records if r.server_id is not None]


class _PairedRawWS:
    """Minimal RawWebSocket: outbound goes to the peer's inbound queue."""

    def __init__(
        self,
        send_q: asyncio.Queue[WSMessage | None],
        recv_q: asyncio.Queue[WSMessage | None],
    ) -> None:
        self._send_q = send_q
        self._recv_q = recv_q
        self.closed = False
        self.close_code: int | None = None

    async def send_bytes(self, data: bytes) -> None:
        await self._send_q.put(WSMessage(WSMsgType.BINARY, data, ""))

    async def close(self) -> bool:
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        return None

    async def close_outbound(self) -> None:
        await self._send_q.put(None)

    async def receive(self) -> WSMessage:
        msg = await self._recv_q.get()
        if msg is None:
            return WSMessage(WSMsgType.CLOSED, None, "")
        return msg


def _paired_encrypted_ws() -> tuple[
    EncryptedWebSocket,
    EncryptedWebSocket,
    _PairedRawWS,
    _PairedRawWS,
]:
    """Return (client_ews, server_ews, client_raw, server_raw) — two transport-mode wrappers.

    The raw ends are handed back so a test can close one side's outbound to
    simulate an early disconnect.
    """
    server_id, client_id = Identity.generate(), Identity.generate()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=server_id.private_bytes,
        remote_static_pub=client_id.public_bytes,
        prologue=b"p",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=client_id.private_bytes,
        remote_static_pub=server_id.public_bytes,
        prologue=b"p",
    )
    msg1 = initiator.write_message(b"")
    responder.read_message(msg1)
    responder.mix_psk(psk)
    initiator.read_message(responder.write_message(b""))

    c2s: asyncio.Queue[WSMessage | None] = asyncio.Queue()
    s2c: asyncio.Queue[WSMessage | None] = asyncio.Queue()
    client_raw = _PairedRawWS(c2s, s2c)
    server_raw = _PairedRawWS(s2c, c2s)
    return (
        EncryptedWebSocket(client_raw, responder),
        EncryptedWebSocket(server_raw, initiator),
        client_raw,
        server_raw,
    )


async def test_pairing_psk_finalize_round_trip() -> None:
    """Both sides persist matching records carrying the client-generated long-term PSK."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(server_ews, client_id="client-A", store=server_store),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    # The same long-term PSK is recorded on both sides.
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    # Directional counterparties.
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    # The server persisted its record too.
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_client_finalize_raises_if_server_closes_before_ack() -> None:
    """If the server closes before sending server/pair-finalize, the client raises."""
    client_ews, _server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    await server_raw.close_outbound()  # server never acks
    with pytest.raises(PairingError, match="closed while awaiting ServerPairFinalizeMessage"):
        await run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        )
    # Nothing persisted on failure (only the pre-provisioned shared record remains).
    assert _added_records(await client_store.records()) == []


async def test_server_finalize_raises_if_client_closes_first() -> None:
    """If the client closes before sending client/pair-finalize, the server raises."""
    _client_ews, server_ews, client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_raw.close_outbound()  # client never sends client/pair-finalize
    with pytest.raises(PairingError, match="closed while awaiting ClientPairFinalizeMessage"):
        await run_pairing_psk_server(server_ews, client_id="client-A", store=server_store)
    assert await server_store.record_by_client_id("client-A") is None


_HANDSHAKE_HASH = bytes(range(32))


async def test_dynamic_pin_round_trip() -> None:
    """A matching PIN authenticates the PAKE and both sides persist the record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide() -> str:
        return await shown  # operator types the PIN the client displayed

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pin_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pin_length=8,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    assert server_record is not None  # a finalized pairing returns a record
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_dynamic_pin_wrong_pin_aborts_and_persists_nothing() -> None:
    """A PIN mismatch fails confirmation; both sides abort and store nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide_wrong() -> str:
        pin = await shown
        wrong_first = "2" if pin[0] == "1" else "1"  # guaranteed different from the shown PIN
        return wrong_first + pin[1:]

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_dynamic_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            run_dynamic_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_length=8,
                pin_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    assert await client_store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 1
    assert _added_records(await client_store.records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pin_length_below_client_floor_aborts() -> None:
    """A pin_length under the client's min is rejected pre-PAKE; the counter is untouched."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()  # default min_pin_length = 6
    server_store = InMemoryServerPairingStore()
    emitted: list[str] = []

    async def emit(pin: str) -> None:
        emitted.append(pin)

    async def provide() -> str:
        return "0000"

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_dynamic_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            run_dynamic_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_length=4,  # below the client's floor of 6
                pin_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_LENGTH_UNACCEPTABLE
    assert emitted == []  # no PIN emitted on a length rejection
    assert await client_store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 0
    assert _added_records(await client_store.records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_client_relays_leave_pairing_without_storing() -> None:
    """A server that leaves pairing makes the client relay the server/activate and store nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pin_failure(PairMethod.DYNAMIC_PIN)  # a prior failure to be reset
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async def server_leaves_pairing() -> None:
        # Receive client/pair-finalize without finalizing, then leave pairing with a
        # server/activate (what the connection layer sends in place of an ack).
        await run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pin_length=8,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
            finalize=False,
        )
        await server_ews.send_str(
            ServerActivateMessage(
                payload=ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[]),
            ).to_json(),
        )

    leftover, _ = await asyncio.gather(
        run_dynamic_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pin_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        server_leaves_pairing(),
    )

    # The client relayed the raw server/activate frame and stored nothing on either side.
    assert leftover is not None
    assert "server/activate" in leftover
    assert _added_records(await client_store.records()) == []
    assert await server_store.record_by_client_id("client-A") is None
    # Inner authentication succeeded, so the failure counter resets like any other attempt.
    assert await client_store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 0


_STATIC_PIN = "12345678"


async def test_static_pin_round_trip() -> None:
    """A matching static PIN authenticates the PAKE; both sides persist and the counter resets."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pin_failure(PairMethod.STATIC_PIN)  # a prior failure to be reset

    async def provide() -> str:
        return _STATIC_PIN

    _client_ret, server_record = await asyncio.gather(
        run_static_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            static_pin=_STATIC_PIN,
            server_id="server-X",
            store=client_store,
        ),
        run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record
    # A successful pairing resets the failure counter.
    assert await client_store.pin_failure_count(PairMethod.STATIC_PIN) == 0


async def test_static_pin_wrong_pin_aborts_and_increments_counter() -> None:
    """A static-PIN mismatch aborts, increments the client counter, and stores nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    async def provide_wrong() -> str:
        return "87654321"

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_static_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                static_pin=_STATIC_PIN,
                server_id="server-X",
                store=client_store,
            ),
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    assert await client_store.pin_failure_count(PairMethod.STATIC_PIN) == 1
    assert _added_records(await client_store.records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_static_pin_low_order_server_share_aborts_cleanly() -> None:
    """A low-order CPace share from the server aborts the client with pin_mismatch."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def malicious_server() -> None:
        await server_ews.receive()  # client/pair-init
        await server_ews.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(bytes(32))),
            ).to_json(),
        )

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_static_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                static_pin=_STATIC_PIN,
                server_id="server-X",
                store=client_store,
            ),
            malicious_server(),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    # A malformed share is not a PIN guess, so the counter is untouched (mirrors the server).
    assert await client_store.pin_failure_count(PairMethod.STATIC_PIN) == 0
    assert _added_records(await client_store.records()) == []


async def test_static_pin_malformed_server_share_raises() -> None:
    """A non-base64 CPace share from the server is a malformed-message error, not a PIN guess."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def malicious_server() -> None:
        await server_ews.receive()  # client/pair-init
        await server_ews.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1="!!!notbase64!!!"),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                static_pin=_STATIC_PIN,
                server_id="server-X",
                store=client_store,
            ),
            malicious_server(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await client_store.pin_failure_count(PairMethod.STATIC_PIN) == 0
    assert _added_records(await client_store.records()) == []


async def test_static_pin_malformed_client_share_raises() -> None:
    """A non-base64 CPace share from the client aborts the server without persisting a record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PIN

    async def malicious_client() -> None:
        await client_ews.send_str(ClientPairInitMessage(payload=ClientPairInitPayload()).to_json())
        await client_ews.receive()  # server/pair-auth
        await client_ews.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2="!!!notbase64!!!"),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pin_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            malicious_client(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


class _ExhaustedClientStore(InMemoryClientPairingStore):
    """A client store that cannot persist new records (to exercise the fallback)."""

    async def can_store_record(self) -> bool:
        return False


async def test_pairing_psk_falls_back_to_shared_when_storage_exhausted() -> None:
    """On storage exhaustion the client hands the server its configured shared PSK.

    No new record is created on the client; the server stores the shared PSK as
    its own long-term record keyed by client_id.
    """
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = _ExhaustedClientStore()
    server_store = InMemoryServerPairingStore()

    shared_psk = generate_psk()
    shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk, server_id=None)
    await client_store.store_record(shared)
    await client_store.set_record_mode_psk_id(shared.psk_id)

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(server_ews, client_id="client-A", store=server_store),
    )

    # The client admitted the server under the shared record: no new stored-pubkey record.
    assert _added_records(await client_store.records()) == []
    assert await client_store.record_by_psk_id(shared.psk_id) is not None
    # The server received and stored the shared PSK.
    assert server_record.psk == shared_psk
    assert await server_store.record_by_client_id("client-A") == server_record
