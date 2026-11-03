"""One-shot process-group supervision with optional scoped Windows wakefulness.

The child command is an argv JSON list, never a shell command. The Windows
helper keeps a thread-scoped execution-state request only while its stdin is
open. No persistent power settings, scheduled tasks, or retries are involved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


POWERSHELL_KEEP_AWAKE = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CodexHartfordExecutionState {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
$previous = [CodexHartfordExecutionState]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) { throw 'SetThreadExecutionState failed' }
try {
    [Console]::Out.WriteLine('READY')
    [Console]::Out.Flush()
    [Console]::In.ReadToEnd() | Out-Null
} finally {
    $cleared = [CodexHartfordExecutionState]::SetThreadExecutionState([uint32]2147483648)
    if ($cleared -eq 0) { throw 'Clearing SetThreadExecutionState failed' }
}
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read physical MemAvailable from /proc/meminfo')


def group_memory(pgid):
    """RSS is diagnostic: shared pages may be counted more than once."""
    members, rss = [], 0
    page_size = os.sysconf('SC_PAGE_SIZE')
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pgid:  # pgrp is /proc stat field 5.
                continue
            members.append(int(directory.name))
            rss += int(fields[21]) * page_size  # rss is field 24.
        except (OSError, ValueError, IndexError):
            continue
    return dict(pids=sorted(members), rss_bytes=rss)


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_owned_group(child, *, grace_seconds):
    """Terminate only the session created for this child, then reap its leader."""
    pgid = child.pid
    if pgid == os.getpgrp():
        raise RuntimeError('Refusing to signal the supervisor process group')
    details = dict(process_group=pgid, term_sent=False, kill_sent=False)
    try:
        os.killpg(pgid, signal.SIGTERM)
        details['term_sent'] = True
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while _group_exists(pgid) and time.monotonic() < deadline:
        child.poll()  # Reap a terminated group leader while descendants exit.
        time.sleep(min(.05, max(0., deadline-time.monotonic())))
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            details['kill_sent'] = True
        except ProcessLookupError:
            pass
    child.wait(timeout=5)
    details['child_exit_code'] = child.returncode
    return details


class SupervisorSignal(Exception):
    def __init__(self, signum):
        super().__init__(f'Supervisor received signal {signum}')
        self.signum = signum


class WindowsAwakeKeeper:
    """A blocking PowerShell thread owns and clears its execution-state flag."""
    def __init__(self, *, executable='powershell.exe', command=None):
        self.command = command or [executable, '-NoLogo', '-NoProfile', '-NonInteractive',
                                   '-Command', POWERSHELL_KEEP_AWAKE]
        self.process = None
        self.error_log = None
        self.ready = False

    def start(self, root, *, timeout, check_signal):
        self.error_log = (Path(root) / 'windows-awake.log').open('ab', buffering=0)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.error_log, start_new_session=True)
        deadline = time.monotonic() + timeout
        received = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                check_signal()
                if self.process.poll() is not None:
                    raise RuntimeError(f'Windows awake helper exited before READY: {self.process.returncode}')
                if not selector.select(min(.1, max(0., deadline-time.monotonic()))):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    raise RuntimeError('Windows awake helper closed stdout before READY')
                received.extend(chunk)
                if len(received) > 16384:
                    raise RuntimeError('Unexpected Windows awake startup output')
                while b'\n' in received:
                    line, _, remainder = received.partition(b'\n')
                    received = bytearray(remainder)
                    if line.strip() == b'READY':
                        if self.process.poll() is not None:
                            raise RuntimeError('Windows awake helper exited during startup')
                        self.ready = True
                        return dict(requested=True, ready=True, pid=self.process.pid)
        raise TimeoutError('Windows awake helper did not report READY before its startup deadline')

    def poll(self):
        return None if self.process is None else self.process.poll()

    def close(self):
        result = dict(ready=self.ready, graceful_release=False, exit_code=None)
        try:
            if self.process is None:
                return result
            if self.process.stdin is not None and not self.process.stdin.closed:
                self.process.stdin.close()  # EOF releases the request in the helper's finally block.
            try:
                self.process.wait(timeout=5)
                result['graceful_release'] = self.ready and self.process.returncode == 0
            except subprocess.TimeoutExpired:
                result['forced_cleanup'] = stop_owned_group(self.process, grace_seconds=1.)
            result['exit_code'] = self.process.returncode
            if self.process.stdout is not None:
                self.process.stdout.close()
            return result
        finally:
            if self.error_log is not None:
                self.error_log.close()


def load_command(path):
    command = json.loads(Path(path).read_text())
    if (not isinstance(command, list) or not command or
            any(not isinstance(item, str) or '\0' in item for item in command) or not command[0]):
        raise ValueError('Command JSON must be a nonempty argv list of strings')
    return command


def supervise(args):
    if os.name != 'posix':
        raise RuntimeError('This supervisor requires POSIX process groups (Linux or WSL)')
    for name in ('min_available_mib', 'memory_grace_seconds', 'check_seconds',
                 'terminate_grace_seconds', 'keeper_startup_seconds'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    command = load_command(args.command)
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Supervisor output already contains evidence; automatic retries are disabled')
    root.mkdir(parents=True, exist_ok=True)
    status = dict(schema=1, status='starting', command=command, command_file=str(args.command.resolve()),
        command_sha256=hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest(),
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cwd=str(Path(args.cwd or Path.cwd()).resolve()), supervisor_pid=os.getpid(), started_at=utc_now(),
        child_pid=None, child_exit_code=None, exit_code=None, guard_exit_code=None, attempts=0,
        limits=dict(min_available_mib=args.min_available_mib, memory_grace_seconds=args.memory_grace_seconds,
                    check_seconds=args.check_seconds, terminate_grace_seconds=args.terminate_grace_seconds),
        windows_awake=dict(requested=args.keep_windows_awake, ready=False))
    status_path = root / 'guard-status.json'
    atomic_json(status_path, status)
    received_signal = None
    old_handlers = {}
    child, keeper = None, None
    failure = None
    low_since = None
    started = time.monotonic()

    def signal_handler(signum, frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    def check_signal():
        if received_signal is not None:
            raise SupervisorSignal(received_signal)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, signal_handler)
        if args.keep_windows_awake:
            keeper = WindowsAwakeKeeper(executable=args.powershell)
            status['windows_awake'] = keeper.start(root, timeout=args.keeper_startup_seconds,
                                                  check_signal=check_signal)
            atomic_json(status_path, status)
        check_signal()
        # Verify the physical-memory metric is readable before starting a child.
        available_memory_bytes()
        with (root / 'runner.log').open('ab', buffering=0) as log, \
             (root / 'host-memory-samples.jsonl').open('a', buffering=1) as samples:
            if keeper is not None and keeper.poll() is not None:
                raise RuntimeError('Windows awake helper exited before child launch')
            child = subprocess.Popen(command, cwd=args.cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status='running', child_pid=child.pid, process_group=child.pid, attempts=1)
            atomic_json(status_path, status)
            while True:
                check_signal()
                if keeper is not None and keeper.poll() is not None:
                    raise RuntimeError(f'Windows awake helper exited unexpectedly: {keeper.poll()}')
                returncode = child.poll()
                if returncode is not None:
                    status.update(status='exited', reason='child_exit', child_exit_code=returncode,
                                  exit_code=returncode, guard_exit_code=returncode if returncode >= 0 else 128-returncode)
                    break
                now = time.monotonic()
                available = available_memory_bytes()
                group = group_memory(child.pid)
                if available < args.min_available_mib * 1024**2:
                    if low_since is None:
                        low_since = now
                else:
                    low_since = None
                sample = dict(timestamp=utc_now(), elapsed_seconds=now-started,
                    available_memory_bytes=available, low_memory_seconds=0. if low_since is None else now-low_since,
                    process_group_rss_bytes=group['rss_bytes'], process_group_pids=group['pids'])
                samples.write(json.dumps(sample, sort_keys=True) + '\n')
                status['last_sample'] = sample
                atomic_json(status_path, status)
                if low_since is not None and now-low_since >= args.memory_grace_seconds:
                    status.update(status='stopping', reason='low_memory', guard_exit_code=75)
                    atomic_json(status_path, status)
                    break
                deadline = now + args.check_seconds
                while time.monotonic() < deadline:
                    check_signal()
                    time.sleep(min(.1, max(0., deadline-time.monotonic())))
    except SupervisorSignal as error:
        failure = error
        status.update(status='stopping', reason='signal', signal=error.signum, guard_exit_code=128+error.signum)
    except KeyboardInterrupt as error:
        failure = error
        status.update(status='stopping', reason='signal', signal=signal.SIGINT, guard_exit_code=130)
    except BaseException as error:
        failure = error
        status.update(status='failed', reason='supervisor_error', error=f'{type(error).__name__}: {error}',
                      guard_exit_code=70)
    finally:
        cleanup_errors = []
        if child is not None:
            try:
                # Also clean up any descendants left behind by a normally exiting leader.
                status['cleanup'] = stop_owned_group(child, grace_seconds=args.terminate_grace_seconds)
                status['child_exit_code'] = status['exit_code'] = child.returncode
            except BaseException as error:
                cleanup_errors.append(f'Child cleanup: {type(error).__name__}: {error}')
        if keeper is not None:
            try:
                released = keeper.close()
                status['windows_awake']['release'] = released
                if status['windows_awake'].get('ready') and not released.get('graceful_release'):
                    cleanup_errors.append('Windows awake helper did not confirm graceful release')
            except BaseException as error:
                cleanup_errors.append(f'Awake-helper cleanup: {type(error).__name__}: {error}')
        if cleanup_errors:
            status.update(cleanup_errors=cleanup_errors, guard_exit_code=70)
            if status['status'] == 'exited':
                status.update(status='failed', reason='cleanup_failure')
        if status['status'] == 'stopping':
            status['status'] = 'stopped'
        if failure is not None:
            status.setdefault('error', f'{type(failure).__name__}: {failure}')
        status.update(finished_at=utc_now(), elapsed_seconds=time.monotonic()-started)
        try:
            atomic_json(status_path, status)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, required=True)
    result.add_argument('--command', type=Path, required=True, help='JSON argv list; no shell expansion')
    result.add_argument('--cwd', type=Path)
    result.add_argument('--min-available-mib', type=float, default=1536)
    result.add_argument('--memory-grace-seconds', type=float, default=5)
    result.add_argument('--check-seconds', type=float, default=2)
    result.add_argument('--terminate-grace-seconds', type=float, default=15)
    result.add_argument('--keep-windows-awake', action='store_true')
    result.add_argument('--powershell', default='powershell.exe')
    result.add_argument('--keeper-startup-seconds', type=float, default=15)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = supervise(args)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', flush=True)
        return 70
    print(json.dumps(result, sort_keys=True), flush=True)
    return result['guard_exit_code'] if result['guard_exit_code'] is not None else 70


if __name__ == '__main__':
    raise SystemExit(main())
