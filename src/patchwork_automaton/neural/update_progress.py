"""Read-only update telemetry and a wall-clock budget checked at work boundaries."""
import math
import time

class UpdateProgress:
    def __init__(self,callback=None,*,timeout_seconds=None,interval_seconds=10):
        if callback is not None and not callable(callback):
            raise ValueError('Update progress callback must be callable')
        if timeout_seconds is not None and (isinstance(timeout_seconds,bool)
                or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError('Update time budget must be positive and finite')
        if not math.isfinite(interval_seconds) or interval_seconds<0:
            raise ValueError('Progress interval must be nonnegative and finite')
        self.callback,self.timeout_seconds=callback,timeout_seconds
        self.interval_seconds=interval_seconds
        self.started=time.monotonic();self.last=self.started
        self.phase=None

    def emit(self,phase,*,force=False,**fields):
        now=time.monotonic();elapsed=now-self.started
        if self.timeout_seconds is not None and elapsed>=self.timeout_seconds:
            if self.callback is not None:
                self.callback(dict(phase='timeout',last_phase=phase,elapsed_seconds=elapsed,
                                   timeout_seconds=self.timeout_seconds))
            raise TimeoutError(f'PPO update exceeded {self.timeout_seconds:g}-second wall-clock budget during {phase}')
        if self.callback is not None and (force or phase!=self.phase or now-self.last>=self.interval_seconds):
            self.callback(dict(phase=phase,elapsed_seconds=elapsed,**fields))
            self.last=now
        self.phase=phase
