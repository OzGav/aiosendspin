"""End-to-end Noise tests: pairing, paired playback, bad PSK, and transition mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestServer

from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ServerActivateMessage,
    ServerActivatePayload,
    ServerHelloMessage,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    ClientMessage,
    ClientStateType,
    PairAbortReason,
    PairMethod,
    PlayerCommand,
    Roles,
    ServerMessage,
    TrustLevel,
)
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.pairing import PairingAbortError, PairingAttempt
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client


def _make_server(
    store: InMemoryServerPairingStore,
    *,
    allow_unencrypted: bool = False,
    min_pin_length: int = 6,
) -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=store,
        allow_unencrypted=allow_unencrypted,
        min_pin_length=min_pin_length,
    )


@asynccontextmanager
async def _serve(server: SendspinServer) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get(SendspinServer.API_PATH, server.on_client_connect)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield f"ws://127.0.0.1:{test_server.port}{SendspinServer.API_PATH}"
    finally:
        await test_server.close()
        await server.close()


def _legacy_hello() -> str:
    return ClientHelloMessage(
        payload=ClientHelloPayload(
            client_id="legacy-client",
            name="legacy",
            version=1,
            supported_roles=[Roles.CONTROLLER.value],
        )
    ).to_json()


async def test_pairing_psk_flow_then_paired_playback() -> None:
    """Pair via a Pairing PSK, then reconnect with the established long-term PSK."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    # Operator-style setup: a Pairing PSK the client accepts and the server stages.
    pairing = generate_psk()
    pp = PairingPsk(psk_id=psk_id_for(pairing), psk=pairing)
    await client_store.set_pairing_psk(pp)
    await server_store.stage_pairing_psk(client_identity.peer_id, pp)

    async with _serve(server) as url:
        pair_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        # Pairing finalizes, the server re-handshakes onto the long-term PSK, and the
        # connection continues as a normal session (no disconnect).
        await pair_client.connect(url)
        assert pair_client.connected
        assert pair_client.noise_psk is not None
        assert pair_client.noise_psk.category is PskCategory.LONG_TERM

        client_record = await client_store.record_by_server_id(server.id)
        server_record = await server_store.record_by_client_id(client_identity.peer_id)
        assert client_record is not None
        assert server_record is not None
        assert client_record.psk == server_record.psk
        assert client_record.psk_id == server_record.psk_id
        await pair_client.disconnect()

        # Reconnect with the long-term PSK for a playback connection.
        play_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await play_client.connect(url)
            assert play_client.connected
            assert play_client.server_info is not None
            assert play_client.server_info.server_id == server.id
            assert play_client.noise_psk is not None
            assert play_client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await play_client.disconnect()


async def test_transition_mode_accepts_legacy_client() -> None:
    """With allow_unencrypted, a legacy client opening with client/hello gets server/hello."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type is WSMsgType.TEXT
        assert isinstance(ServerMessage.from_json(msg.data), ServerHelloMessage)


async def test_default_server_rejects_legacy_client() -> None:
    """Without transition mode, a legacy client/hello is closed without a server/hello."""
    server = _make_server(InMemoryServerPairingStore())
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)


async def _find_connection_by_client_id(
    server: SendspinServer, client_id: str
) -> SendspinConnection:
    async with asyncio.timeout(5):
        while True:
            for conn in server._pending_connections:  # noqa: SLF001
                if conn._client_id == client_id:  # noqa: SLF001
                    return conn
            await asyncio.sleep(0.01)


async def _await_long_term_record(store: InMemoryClientPairingStore, server_id: str) -> None:
    async with asyncio.timeout(5):
        while await store.record_by_server_id(server_id) is None:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_unknown_client_admitted_idle_on_sentinel() -> None:
    """An unknown client lands on Sentinel and receives server/activate(activities=[])."""
    server = _make_server(InMemoryServerPairingStore())
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=Identity.generate(),
            pairing_store=InMemoryClientPairingStore(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.SENTINEL
            assert client.activities == []
        finally:
            await client.disconnect()


async def test_live_pairing_dynamic_pin() -> None:
    """Operator pairs a Sentinel-idle connection via Dynamic PIN."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pin_display=display,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
            assert client_record.psk_id == server_record.psk_id
            # Both floors default to 6, so the negotiated PIN is 6 digits.
            assert len(shown.result()) == 6
        finally:
            await client.disconnect()


async def test_live_pairing_dynamic_pin_server_floor_raises_length() -> None:
    """The negotiated length is max(client_min, server_min); the server's higher floor wins."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store, min_pin_length=8)  # client default floor is 6
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pin_display=display,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
            assert len(shown.result()) == 8
        finally:
            await client.disconnect()


async def test_live_pairing_pairing_psk() -> None:
    """Operator pairs a Sentinel-idle connection via Pairing PSK."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=pairing)
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
        finally:
            await client.disconnect()


async def test_live_pairing_static_pin() -> None:
    """Operator pairs a Sentinel-idle connection via Static PIN once the window opens."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), static_pin_enabled=True)
    )
    await client_store.set_static_pin("12345678")
    await client_store.record_pin_failure(PairMethod.STATIC_PIN)  # reset on success

    window_opened = asyncio.get_running_loop().create_future()

    async def open_window() -> None:
        if not window_opened.done():
            window_opened.set_result(None)

    async def provide() -> str:
        return "12345678"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_window=open_window,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.STATIC_PIN, pin_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
            assert window_opened.done()
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
            assert client_record.psk_id == server_record.psk_id
            assert await client_store.pin_failure_count(PairMethod.STATIC_PIN) == 0
        finally:
            await client.disconnect()


async def test_live_pairing_static_pin_locked_out_aborts() -> None:
    """A static-PIN attempt under terminal lockout aborts and persists no record."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), static_pin_enabled=True)
    )
    await client_store.set_static_pin("12345678")
    for _ in range(10):
        await client_store.record_pin_failure(PairMethod.STATIC_PIN)
    assert await client_store.is_pin_locked_out(PairMethod.STATIC_PIN)

    async def open_window() -> None:
        return

    async def provide() -> str:
        return "12345678"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_window=open_window,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingAbortError) as excinfo:
                await conn.initiate_pairing(
                    PairingAttempt(method=PairMethod.STATIC_PIN, pin_provider=provide)
                )
            assert excinfo.value.reason is PairAbortReason.LOCKED_OUT
            assert await client_store.record_by_server_id(server.id) is None
        finally:
            await client.disconnect()


async def test_live_pairing_pauses_writer_during_exchange() -> None:
    """initiate_pairing pauses the writer for the duration of the pairing exchange.

    The pairing/re-handshake sends and the writer share one Noise send-cipher, so a
    writer frame interleaved with them would advance the cipher nonce out of order and
    break the session. The pause is set in initiate_pairing for every method, so the
    dynamic-PIN flow here exercises it for all of them.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()
    writer_paused_mid_exchange: asyncio.Future[bool] = loop.create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pin_display=display,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)

            async def provide() -> str:
                # Mid-exchange: server/pair-init is out and the server awaits the PIN.
                if not writer_paused_mid_exchange.done():
                    writer_paused_mid_exchange.set_result(conn._writer_task is None)  # noqa: SLF001
                # Queue writer work; with the writer paused it must wait for resume
                # rather than interleave with the rest of the exchange.
                for _ in range(64):
                    conn.send_priority_message(
                        ServerActivateMessage(payload=ServerActivatePayload(activities=[]))
                    )
                return await shown

            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide)
            )

            assert await writer_paused_mid_exchange, "writer ran during the pairing exchange"
            assert conn._writer_task is not None  # noqa: SLF001  # resumed after the exchange
            await _await_long_term_record(client_store, server.id)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


async def test_live_pairing_psk_pauses_writer_across_rehandshakes() -> None:
    """Pairing-PSK live pairing re-handshakes twice (Sentinel→Pairing→long-term).

    The writer stays paused across both re-handshakes, so it cannot interleave with
    either one. The store_record hook observes the writer state mid-exchange (after the
    first re-handshake, before the second).
    """
    loop = asyncio.get_running_loop()
    writer_paused_mid_exchange: asyncio.Future[bool] = loop.create_future()
    conn_holder: list[SendspinConnection] = []

    class _ObservingStore(InMemoryServerPairingStore):
        async def store_record(self, record: ServerPairingRecord) -> None:
            if conn_holder and not writer_paused_mid_exchange.done():
                writer_paused_mid_exchange.set_result(conn_holder[0]._writer_task is None)  # noqa: SLF001
            await super().store_record(record)

    server_store = _ObservingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            conn_holder.append(conn)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=pairing)
            )
            assert await writer_paused_mid_exchange, "writer ran during the pairing exchange"
            assert conn._writer_task is not None  # noqa: SLF001  # resumed after the exchange
            await _await_long_term_record(client_store, server.id)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


async def _await_player_state(conn: SendspinConnection, *, volume: int, muted: bool) -> None:
    async with asyncio.timeout(5):
        while True:
            server_client = conn._client  # noqa: SLF001
            if server_client is not None:
                for role in server_client.active_roles:
                    if (
                        role.role_family == "player"
                        and role.get_player_volume() == volume
                        and role.get_player_muted() == muted
                    ):
                        return
            await asyncio.sleep(0.01)


async def test_resync_resends_current_player_state() -> None:
    """After a re-verification, the client re-pushes its *current* player state.

    Dynamic PIN over the long-term PSK re-verifies the pairing: the channel stays on the
    long-term PSK (no re-handshake), and the leave-pairing server/activate reactivates the
    player role. The client follows it with a fresh client/state carrying the volume/mute it
    last reported, not the construction-time initial values.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    # Pre-stage a shared long-term PSK so the client connects directly as paired playback.
    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id)
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async def provide() -> str:
        return await shown

    player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16)
        ],
        buffer_capacity=1_000_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=player_support,
            pin_display=display,
        )
        try:
            await client.connect(url)
            assert client.connected
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)

            # The app moves volume/mute off the initial defaults (100/False).
            await client.send_player_state(
                state=ClientStateType.SYNCHRONIZED, volume=42, muted=True
            )
            await _await_player_state(conn, volume=42, muted=True)

            resync_state: asyncio.Future[tuple[int | None, bool | None]] = loop.create_future()
            pairing_started = False
            original_handle = conn._handle_message  # noqa: SLF001

            async def spy(message: ClientMessage, timestamp_us: int) -> None:
                if (
                    pairing_started
                    and isinstance(message, ClientStateMessage)
                    and message.payload.player is not None
                    and not resync_state.done()
                ):
                    resync_state.set_result(
                        (message.payload.player.volume, message.payload.player.muted)
                    )
                await original_handle(message, timestamp_us)

            conn._handle_message = spy  # type: ignore[method-assign]  # noqa: SLF001

            pairing_started = True
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide)
            )

            async with asyncio.timeout(5):
                resent = await resync_state
            assert resent == (42, True)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


async def _await_connected_client(server: SendspinServer, client_id: str) -> SendspinClient:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is not None and client.is_connected:
                return client
            await asyncio.sleep(0.01)


async def test_reverification_over_long_term_keeps_pairing() -> None:
    """Dynamic PIN over a long-term PSK re-verifies without disturbing the pairing.

    The server runs the dynamic-PIN PAKE round but leaves pairing instead of finalizing: the
    connection stays on the *same* long-term PSK, no new record is stored on either side, and
    roles are reactivated. A successful round resets the failure counter like any other attempt.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id)
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    # A pre-existing dynamic-PIN failure count is reset by a successful re-verification.
    await client_store.record_pin_failure(PairMethod.DYNAMIC_PIN)

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pin_display=display,
        )
        try:
            await client.connect(url)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            server_client = await _await_connected_client(server, client_identity.peer_id)
            # The accessor reports the pre-check security state.
            assert server_client.is_paired
            security = server_client.connection_security
            assert security is not None
            assert security.psk_category is PskCategory.LONG_TERM
            assert security.trust_level is TrustLevel.USER

            await server.initiate_pairing(
                client_identity.peer_id,
                PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide),
            )

            # The connection survives and stays on the *same* long-term PSK.
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert client.noise_psk.psk == long_term
            assert server_client.is_paired

            # No new record on either side; the original pairing is intact.
            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == long_term
            assert server_record.psk == long_term
            # The seeded long-term record plus the pre-provisioned shared fallback; nothing new.
            stored_pubkey = [r for r in await client_store.records() if r.server_id is not None]
            assert len(stored_pubkey) == 1
            # Inner authentication succeeded, so the failure counter resets to zero.
            assert await client_store.pin_failure_count(PairMethod.DYNAMIC_PIN) == 0
        finally:
            await client.disconnect()


async def test_reverification_rejected_under_dynamic_pin_lockout() -> None:
    """Re-verification is subject to lockout: under terminal lockout it aborts with locked_out.

    Spec :pin-pairing-lockout — re-verification follows the lockout rules like any other attempt.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id)
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    # Drive dynamic PIN into terminal lockout (counter reaches 10).
    for _ in range(10):
        await client_store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    assert await client_store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pin: str | None) -> None:
        if pin is not None and not shown.done():
            shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pin_display=display,
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingAbortError) as excinfo:
                await conn.initiate_pairing(
                    PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=provide)
                )
            assert excinfo.value.reason is PairAbortReason.LOCKED_OUT
        finally:
            await client.disconnect()


async def test_initiate_pairing_raises_when_client_not_connected() -> None:
    """The server-level wrapper rejects a presence/pairing request for an absent client."""
    server = _make_server(InMemoryServerPairingStore())
    async with _serve(server):
        with pytest.raises(ValueError, match="not connected"):
            await server.initiate_pairing(
                "unknown-client",
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk()),
            )


async def test_connection_security_reports_sentinel_for_unpaired() -> None:
    """An unpaired (Sentinel) connection reports is_paired=False and trust none."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=InMemoryClientPairingStore(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            server_client = await _await_connected_client(server, identity.peer_id)
            assert not server_client.is_paired
            security = server_client.connection_security
            assert security is not None
            assert security.psk_category is PskCategory.SENTINEL
            assert security.trust_level is TrustLevel.NONE
        finally:
            await client.disconnect()
