"""Minimal authenticated idle session. Does not join games or trade."""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import secrets
import signal
import ssl
import struct
import sys
import time
import uuid

from patchwork_automaton.protocol import Catalog, Cursor, ProtocolError, array, current_time, login_payload, login_response, read_frame
from patchwork_automaton.game import decks_response, solitaire_payload, game_status, event_sync, count
from patchwork_automaton.state import State
from .policy import SoloPolicy, actions, card_payload

RESPONSES = {
    0: "NotSet", 1: "Success", 2: "UnknownError", 3: "PermissionDenied",
    4: "SystemBusy", 5: "UnrecoverableError", 6: "Require2FAChallenge",
    7: "Incorrect2FAChallenge", 100: "UserIsBanned", 101: "InvalidCredentials",
    102: "UserIsDeactivated", 103: "ServerIsInMaintenance", 104: "NoLoginsAllowed",
    105: "UserNotInActiveState", 106: "ServerIsFull", 107: "ClientAccessDenied",
    108: "ClientMd5SumIsInvalid", 109: "DuplicateUser",
}


def credentials(path: Path):
    # Input stays outside the repo. Never print its values or exception text.
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        separators = [(line.index(s), s) for s in ("=", ":") if s in line]
        if not separators:
            raise ValueError("Unlabelled credential")
        separator = min(separators)[1]
        key, value = line.split(separator, 1)
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values[key.strip().lower()] = value
    username, password = values["username"], values["password"]
    if not username or not password:
        raise ValueError("Empty credentials")
    return username, password


class Session:
    def __init__(self, args):
        self.args = args
        self.catalog = Catalog(args.catalog)
        self.run_dir = args.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stop = asyncio.Event()
        self.writer = None
        self.ack_mode = False
        self.pending = collections.OrderedDict()
        self.next_id = (secrets.randbits(31) << 1) or 2
        self.counts = collections.Counter()
        self.solitaire_token = None
        self.match_id = None
        self.game_id = None
        string_path = self.run_dir / 'strings.json'
        self.game_state = State(json.loads(string_path.read_text()) if string_path.exists() else ())
        self.game_route = None
        self.last_played = None
        self.bot_actions = 0
        info_path=self.run_dir/'card-info.json'
        self.policy=SoloPolicy(json.loads(info_path.read_text()) if info_path.exists() else {})
        self.state = {"pid": os.getpid(), "connected": False, "phase": "starting",
                      "host": args.host, "port": args.port, "started_at": time.time(),
                      "heartbeats_answered": 0, "time_syncs_answered": 0}

    def publish(self):
        self.state["updated_at"] = time.time()
        self.state["received_counts"] = dict(self.counts)
        self.state["pending_acknowledgements"] = len(self.pending)
        path = self.run_dir / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2))
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def event(self, event, **fields):
        # Only explicitly chosen operational metadata reaches the log.
        print(json.dumps({"time": time.time(), "event": event, **fields}), flush=True)

    async def write(self, frame):
        self.writer.write(frame)
        await self.writer.drain()

    async def send(self, name, payload=b""):
        frame = self.catalog.frame(name, payload)
        if self.ack_mode:
            if len(self.pending) >= 4096:
                raise ProtocolError("Acknowledgement queue exhausted")
            message_id = self.next_id
            self.next_id = ((message_id + 1) & 0xffffffff) or 1
            self.pending[message_id] = time.monotonic()
            frame = self.catalog.frame("AckMessageReqMessage", struct.pack("<I", message_id) + array(frame))
        await self.write(frame)

    async def unwrap(self, frame):
        for _ in range(8):
            opcode, payload = self.catalog.parse(frame)
            entry = self.catalog.by_opcode.get(opcode)
            name = entry["name"] if entry else f"opcode_{opcode}"
            self.counts[name] += 1
            if opcode == 16:
                c = Cursor(payload)
                message_id = struct.unpack("<I", c.take(4))[0]
                frame = c.blob()
                c.end()
                # ACK responses themselves are not wrapped in another ACK.
                await self.write(self.catalog.frame("AckMessageRespMessage", struct.pack("<I", message_id)))
            elif opcode == 17:
                if len(payload) != 4:
                    raise ProtocolError("Invalid acknowledgement length")
                message_id = struct.unpack("<I", payload)[0]
                if message_id in self.pending:
                    while self.pending:
                        key, _ = self.pending.popitem(last=False)
                        if key == message_id:
                            break
                return None
            elif opcode == 18:
                c = Cursor(payload)
                frame = c.blob()
                c.end()
            else:
                return opcode, payload
        raise ProtocolError("Envelope nesting limit exceeded")

    async def handle(self, item):
        if item is None:
            return
        opcode, payload = item
        if opcode in (960, 955) and self.solitaire_token and payload[:16] == self.solitaire_token.bytes_le:
            c = Cursor(payload[16:])
            event = event_sync(c)
            c.end()
            await self.sync_solitaire(event)
        elif opcode == 983:
            c = Cursor(payload)
            events = [event_sync(c) for _ in range(count(c))]
            c.end()
            target_path = self.run_dir / "game-target.json"
            if target_path.exists():
                target = json.loads(target_path.read_text())["match_id"]
                for event in events:
                    if event["event_id"] == target:
                        self.solitaire_token = event["token"]
                        await self.sync_solitaire(event)
        elif opcode in (962, 107) and self.solitaire_token and payload[:16] == self.solitaire_token.bytes_le:
            self.state.update(game_phase="creation_rejected", game_creation_reason=struct.unpack_from("<i", payload, 16)[0])
            self.event("solitaire_creation_rejected", reason=self.state["game_creation_reason"])
        elif opcode in (944, 945) and self.solitaire_token and payload[:16] == self.solitaire_token.bytes_le:
            self.state["game_phase"] = "insufficient_cards" if opcode == 944 else "invalid_deck"
            self.event("solitaire_deck_rejected", reason=self.state["game_phase"])
        elif opcode == 2463 and self.solitaire_token and payload[:16] == self.solitaire_token.bytes_le:
            c = Cursor(payload)
            self.match_token = uuid.UUID(bytes_le=c.take(16))
            self.match_id, self.game_id = c.integer(), c.integer()
            c.end()
            self.state.update(game_phase="game_started", match_id=self.match_id, game_id=self.game_id)
            self.event("solitaire_game_started", match_id=self.match_id, game_id=self.game_id)
            await self.send("GsConnectMessage", self.match_token.bytes_le + struct.pack("<ii", self.match_id, self.game_id))
        elif opcode == 1153 and self.game_id:
            c = Cursor(payload)
            route_token = c.take(16)
            match_id, game_id = c.integer(), c.integer()
            message = c.blob()
            c.end()
            if (match_id, game_id) == (self.match_id, self.game_id):
                self.game_route = route_token
                await self.handle(self.catalog.parse(message))
        elif opcode == 4652 and self.game_id:
            status = game_status(payload)
            if status["game_id"] == self.game_id:
                self.state["game_phase"] = "receiving_game_state"
                self.state["game_status"] = status
                self.state["game_state_messages"] = self.state.get("game_state_messages", 0) + 1
                path = self.run_dir / "game-state.latest.bin"
                path.write_bytes(payload)
                os.chmod(path, 0o600)
                self.event("solitaire_game_state", **status)
                try:
                    complete = self.game_state.update(payload)
                except ProtocolError as error:
                    self.state['state_decoder_error'] = str(error)
                    self.event('state_decode_paused',reason=str(error))
                    (self.run_dir/'state.latest.raw').write_bytes(self.game_state.raw)
                    self.publish()
                    return
                (self.run_dir/'state.latest.raw').write_bytes(self.game_state.raw)
                if complete:
                    self.state.pop('state_decoder_error',None)
                    self.state['prompt'] = self.game_state.prompt
                    path = self.run_dir / 'observation.json'
                    path.write_text(json.dumps({'prompt':self.game_state.prompt,
                        'cards':list(self.game_state.cards.values()), 'players':self.game_state.players},indent=2))
                    os.chmod(path,0o600)
        elif opcode == 967 and self.game_id and len(payload)>=32:
            match_id,game_id,game_status_code,match_state = struct.unpack_from('<4i',payload,16)
            if (match_id,game_id)==(self.match_id,self.game_id):
                self.state.update(game_phase='game_ended',final_game_status=game_status_code,final_match_state=match_state)
                self.event('solitaire_game_ended',game_status=game_status_code)
        if opcode == 94:
            response, decks = decks_response(payload)
            path = self.run_dir / "decks.json"
            path.write_text(json.dumps({"response_code": response, "decks": decks}, indent=2))
            os.chmod(path, 0o600)
            self.state["decks_received"] = len(decks)
            self.state["deck_list_response"] = response
            self.event("deck_list_received", count=len(decks), response_code=response)
        if opcode == 528:
            if len(payload) != 4:
                raise ProtocolError("Invalid heartbeat length")
            await self.send("HeartbeatAckMessage", payload)
            self.state["heartbeats_answered"] += 1
            self.state["last_heartbeat_at"] = time.time()
        elif opcode == 3861:
            if len(payload) != 9:
                raise ProtocolError("Invalid time sync length")
            await self.send("ServerTimeSyncRespMessage", payload + current_time())
            self.state["time_syncs_answered"] += 1
        elif opcode in self.terminal_opcodes:
            self.state["terminal_message"] = self.catalog.by_opcode[opcode]["name"]
            raise ProtocolError("Server ended the session")

    async def send_game(self,name,payload):
        if self.game_route is None or self.game_id is None:
            raise ProtocolError('No connected game route')
        inner = self.catalog.frame(name,payload)
        await self.send('GsMessageMessage',self.game_route+struct.pack('<ii',self.match_id,self.game_id)+array(inner))

    async def play_step(self):
        switch=self.run_dir/'play.json'
        if not switch.exists() or not json.loads(switch.read_text()).get('enabled'):
            return
        if self.state.get('game_phase')!='receiving_game_state' or self.state.get('state_decoder_error'):
            return
        prompt=self.game_state.prompt
        if prompt is None:
            return
        signature=(prompt['timestamp'],hash(self.game_state.raw))
        if signature==self.last_played:
            return
        if self.bot_actions>=300:
            self.state['bot_phase']='action_limit_reached'
            return
        choice=self.policy.choose(self.game_state)
        if choice is None:
            self.state['bot_phase']='unsupported_prompt'
            return
        kind,first,second,label=choice
        if kind=='prompt':
            await self.send_game('PromptActionMessage',struct.pack('<III',first,second,prompt['timestamp']))
        else:
            await self.send_game('NewCardActionMessage',card_payload(first,second,prompt['timestamp']))
        self.last_played=signature
        self.bot_actions+=1
        self.state.update(bot_phase='playing',bot_actions=self.bot_actions,last_bot_choice=label)
        self.event('bot_choice',label=label,turn=prompt['turn'],timestamp=prompt['timestamp'],action_number=self.bot_actions)

    async def sync_solitaire(self, event):
        if event["player_format"] != "SOLO" or event["maximum_players"] != 1:
            raise ProtocolError("Expected a one-player solitaire event")
        self.match_token, self.match_id = event["token"], event["event_id"]
        self.state.update(game_phase="match_created", match_id=self.match_id,
                          match_state=event["state"], maximum_players=event["maximum_players"],
                          game_count=len(event["games"]))
        target = self.run_dir / "game-target.json"
        target.write_text(json.dumps({"match_id": self.match_id}))
        self.event("solitaire_match_synced", match_id=self.match_id, state=event["state"], games=event["games"])
        if event["games"]:
            self.game_id = event["games"][-1]["game_id"]
            self.state.update(game_phase="connecting_to_game", game_id=self.game_id)
            await self.send("GsConnectMessage", self.match_token.bytes_le + struct.pack("<ii", self.match_id, self.game_id))
        elif event["state"] == 3:
            await self.send("FlsMatchHostStartMessage", self.match_token.bytes_le + struct.pack("<i", self.match_id))

    async def connect(self):
        username, password = credentials(self.args.credentials)
        device_path = self.run_dir / "device-id"
        if not device_path.exists():
            device_path.write_text(secrets.token_hex(32).upper())
            os.chmod(device_path, 0o600)
        # A stable ID for this Linux research client, not another device's ID.
        device = device_path.read_text().strip()
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.state["phase"] = "connecting"
        self.publish()
        reader, self.writer = await asyncio.wait_for(asyncio.open_connection(
            self.args.host, self.args.port, ssl=context, server_hostname=self.args.host), 20)
        self.state["tls_version"] = self.writer.get_extra_info("ssl_object").version()
        self.event("tls_connected", tls_version=self.state["tls_version"])
        challenge = self.args.challenge_file.read_text().strip() if self.args.challenge_file else ""
        payload = login_payload(username, password, device, str(uuid.uuid4()), challenge)
        await self.write(self.catalog.frame("FlsUserLoginReqMessage", payload))
        del username, password, payload, challenge
        self.state["phase"] = "authenticating"
        self.publish()
        deadline = time.monotonic() + 30
        while True:
            frame = await asyncio.wait_for(read_frame(reader), max(0.1, deadline - time.monotonic()))
            self.state["last_received_at"] = time.time()
            item = await self.unwrap(frame)
            if item is None:
                continue
            opcode, payload = item
            if opcode != 512:
                await self.handle(item)
                continue
            response = login_response(payload)
            self.state["login_response"] = RESPONSES.get(response.response_code, "UnknownResponse")
            self.state["login_response_code"] = response.response_code
            self.state["challenge_type"] = {0: "NotSet", 1: "Email", 2: "Authenticator"}.get(response.authorization_type, "Unknown")
            self.event("login_response", result=self.state["login_response"], challenge_type=self.state["challenge_type"])
            if response.response_code != 1:
                self.state["phase"] = "challenge_required" if response.response_code == 6 else "login_refused"
                self.publish()
                return
            self.ack_mode = bool(response.client_flags & 0x1000000)
            self.user_id = response.user_id
            await self.send("ServerTimeSyncRespMessage", response.server_time + current_time())
            self.state["time_syncs_answered"] += 1
            await self.send("DosUserroleGetprivilegesMessage", struct.pack("<ii", response.user_id, response.user_id))
            self.state.update(connected=True, phase="authenticated", authenticated_at=time.time(), acknowledgement_mode=self.ack_mode)
            if (self.run_dir / "game-target.json").exists():
                await self.send("FlsPlayerEventCurrentEventListReqMessage")
            self.publish()
            self.event("authenticated", acknowledgement_mode=self.ack_mode)
            break
        while not self.stop.is_set():
            item = await self.unwrap(await asyncio.wait_for(read_frame(reader), 120))
            self.state["last_received_at"] = time.time()
            await self.handle(item)

    async def health(self):
        while not self.stop.is_set():
            if (self.run_dir / "stop").exists():
                self.stop.set()
            command_path = self.run_dir / "command.json"
            if self.state["connected"] and command_path.exists():
                command = json.loads(command_path.read_text())
                command_path.unlink()
                if command.get("command") == "list_decks":
                    await self.send("FlsNetDeckListGetReqMessage", struct.pack("<i", 0))
                    self.event("deck_list_requested")
                elif command.get("command") == "solitaire":
                    if self.solitaire_token:
                        self.event("command_rejected", reason="solitaire_already_requested")
                    else:
                        decks = json.loads((self.run_dir / "decks.json").read_text())["decks"]
                        deck = next(d for d in decks if d["deck_id"] == command["deck_id"])
                        game_structure = deck["description"].split("?")[1]
                        self.solitaire_token = uuid.uuid4()
                        payload = solitaire_payload(self.solitaire_token, deck["cards"], game_structure)
                        await self.send("FlsGameCreateAndJoinMessage", payload)
                        self.state.update(game_phase="creation_requested", deck_id=deck["deck_id"])
                        self.event("solitaire_creation_requested", format=game_structure,
                                   main_deck_cards=sum(c["quantity"] for c in deck["cards"] if not c["sideboard"]))
                elif command.get('command')=='prompt':
                    prompt = self.game_state.prompt
                    code = command['code']
                    if prompt is None or command['timestamp']!=prompt['timestamp'] or not any(b['code']==code and b['enabled'] for b in prompt['buttons']):
                        self.event('command_rejected',reason='stale_or_unavailable_prompt')
                    else:
                        await self.send_game('PromptActionMessage',struct.pack('<III',code,command.get('parameter',0),prompt['timestamp']))
                        self.event('prompt_choice_sent',code=code,timestamp=prompt['timestamp'])
            self.publish()
            if self.state['connected']:
                await self.play_step()
            await asyncio.sleep(1)

    async def run(self):
        exit_code = 0
        self.terminal_opcodes = {self.catalog.by_name[name]["opcode"] for name in
                                 ("UserKickedMessage", "ServerDisconnectMessage")}
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        health = asyncio.create_task(self.health())
        connection = asyncio.create_task(self.connect())
        stopping = asyncio.create_task(self.stop.wait())
        try:
            done, _ = await asyncio.wait((connection, stopping, health), return_when=asyncio.FIRST_COMPLETED)
            if health in done:
                await health
            if connection in done:
                await connection
                if self.state["phase"] in ("challenge_required", "login_refused"):
                    exit_code = 2
        except Exception as error:
            # A service may retry transport failures, but never auth refusals or
            # incompatible protocol responses. Avoid account lockout loops.
            exit_code = 1 if isinstance(error, (ConnectionError, TimeoutError,
                         asyncio.IncompleteReadError, OSError)) else 2
            self.state.update(phase="failed", error_type=type(error).__name__)
            if isinstance(error, ProtocolError):
                self.state["protocol_error"] = str(error)
            self.event("session_failed", error_type=type(error).__name__)
        finally:
            if self.stop.is_set() and self.state["connected"] and self.writer:
                try:
                    await asyncio.wait_for(self.send("UserLogoffMessage"), 3)
                except Exception:
                    pass
            for task in (connection, health, stopping):
                task.cancel()
            await asyncio.gather(connection, health, stopping, return_exceptions=True)
            if self.writer:
                self.writer.close()
                try:
                    await asyncio.wait_for(self.writer.wait_closed(), 3)
                except Exception:
                    pass
            self.state["connected"] = False
            if self.state["phase"] == "authenticated":
                self.state["phase"] = "stopped"
            self.publish()
        return exit_code


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=repo / "docs/discovery/mtgo-message-catalog.json")
    parser.add_argument("--run-dir", type=Path, default=repo / ".local")
    parser.add_argument("--host", default="mtgologin1.mtgo.com")
    parser.add_argument("--port", type=int, default=7770)
    parser.add_argument("--challenge-file", type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(Session(args).run()))


if __name__ == "__main__":
    main()
