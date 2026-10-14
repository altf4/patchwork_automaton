"""Failures callers can handle without inspecting transport internals."""


class Disconnected(ConnectionError):
    pass


class AuthenticationError(Exception):
    def __init__(self, code: int, challenge_type: int):
        self.code, self.challenge_type = code, challenge_type
        super().__init__(f'Login refused (code {code}, challenge type {challenge_type})')


class RequestRejected(Exception):
    def __init__(self, opcode: int, reason: int | None = None):
        self.opcode, self.reason = opcode, reason
        super().__init__(f'Request rejected (opcode {opcode}, reason {reason})')


class StaleDecision(ValueError):
    pass


class UnsupportedInteraction(ValueError):
    pass


class EventOverflow(Disconnected):
    """Consumer fell behind; reconnect explicitly rather than lose observations."""
