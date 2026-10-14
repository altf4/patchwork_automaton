"""Install/manage the pinned, loopback-only XMage integration environment.

No system packages, accounts, CI, or MTGO connections are changed. Downloads and
runtime data live under .local/xmage (ignored by git). Requires Python 3.12+ for
safe tar extraction. `start` leaves the server running until explicit `stop`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.local/xmage'
ASSETS = {
    'mage-full.zip': ('https://github.com/magefree/mage/releases/download/xmage_1.4.62V1/mage-full_1.4.62-dev_2026-10-03_20-47.zip',
                      'd256251ae41155638233c2cb20a7fec82f3f2dea7a468976691664c55e357b16', 'official'),
    'nexus-jre-linux-x64.tar.gz': ('https://github.com/Adamychen/xmage-nexus/releases/download/v0.4.7/nexus-jre-linux-x64.tar.gz',
                                 '526f9dfb28f172f0fbb7c223988f0fc622b313a8b56fa1c8d54420d89000dd22', 'java'),
    'nexus-proxy-linux-x64.tar.gz': ('https://github.com/Adamychen/xmage-nexus/releases/download/v0.4.7/nexus-proxy-linux-x64.tar.gz',
                                   'ab2dc27987b27d3750551051dd012f33a90f6cc7915cb0049c73380793fee6d7', 'proxy'),
}


def install():
    if os.uname().machine != 'x86_64':
        raise SystemExit('Pinned runtime is Linux x86_64; use a matching Java 17 runtime on other platforms.')
    downloads = BASE / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    for name,(url,expected,directory) in ASSETS.items():
        path = downloads/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            print('Downloading',name,flush=True)
            urllib.request.urlretrieve(url,temporary)
            temporary.replace(path)
        with path.open('rb') as archive_file:
            digest = hashlib.file_digest(archive_file,'sha256').hexdigest()
        if digest != expected:
            raise SystemExit(f'Checksum mismatch: {name}; remove the bad archive and retry.')
        out = BASE/directory
        if not out.exists():
            if name.endswith('.zip'):
                with zipfile.ZipFile(path) as archive:
                    if any(not (out / n).resolve().is_relative_to(out.resolve()) for n in archive.namelist()):
                        raise SystemExit('Unsafe zip member')
                    archive.extractall(out)
            else:
                with tarfile.open(path) as archive:
                    archive.extractall(out,filter='data')
    config = BASE/'official/xmage/mage-server/config/config.xml'
    text = config.read_text().replace('serverAddress="0.0.0.0"','serverAddress="127.0.0.1"')
    text = text.replace('secondaryBindPort="-1"','secondaryBindPort="17172"')
    config.write_text(text)
    (BASE/'installed.json').write_text(json.dumps({'assets':ASSETS,'server':'1.4.62-V1','proxy':'v0.4.7'},indent=2))
    print('Installed official XMage 1.4.62-V1 and Nexus proxy v0.4.7.')


def pids():
    path = BASE/'pids.json'
    return json.loads(path.read_text()) if path.exists() else {}


def alive(name,pid):
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        needle = b'mage-server-' if name == 'server' else b'mage-proxy-'
        return str(BASE/'java/bin/java').encode() in cmd and needle in cmd
    except (OSError,ValueError):
        return False


def ready():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8788/ready',timeout=2) as response:
            return json.load(response).get('ready',False)
    except (OSError,ValueError):
        return False


def server_listening():
    # Inspect current kernel state, not a possibly stale startup log.
    try:
        return any(row.split()[1] == '0100007F:4313' and row.split()[3] == '0A'
                   for row in Path('/proc/net/tcp').read_text().splitlines()[1:])
    except (OSError, IndexError):
        return False


def start():
    java = BASE/'java/bin/java'
    if not java.exists():
        raise SystemExit('Run install first.')
    common = [str(java),'-Djava.awt.headless=true']
    common += [f'--add-opens=java.base/{p}=ALL-UNNAMED' for p in ('java.io','java.util','java.lang','java.lang.reflect','java.text')]
    commands = {
        'server': (BASE/'official/xmage/mage-server',common+['-Xmx1024m','-jar','lib/mage-server-1.4.62.jar']),
        'proxy': (BASE/'proxy',common+['-Xmx768m','-jar','mage-proxy-1.4.62.jar','--host','127.0.0.1','--port','17171',
            '--bind','127.0.0.1','--wsPort','8787','--httpPort','8788','--graceSecs','1','--relinkSecs','1','--restoreIds','none']),
    }
    running = pids()
    for name,(cwd,cmd) in commands.items():
        if alive(name,running.get(name,0)):
            continue
        with (BASE/f'{name}.log').open('ab') as log:
            process = subprocess.Popen(cmd,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        running[name] = process.pid
        (BASE/'pids.json').write_text(json.dumps(running))
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        if not all(alive(name,pid) for name,pid in running.items()):
            raise SystemExit('A process exited; inspect .local/xmage/server.log and proxy.log.')
        if ready() and server_listening():
            status();return
        time.sleep(1)
    raise SystemExit('Startup timed out; processes retained for log inspection.')


def status():
    print(json.dumps({'processes':{n:{'pid':p,'running':alive(n,p)} for n,p in pids().items()},
        'proxy_ready':ready(),'server_listening':server_listening(),'server':'127.0.0.1:17171','proxy':'ws://127.0.0.1:8787',
        'health':'http://127.0.0.1:8788/ready','logs':str(BASE)},indent=2))


def stop():
    running = pids()
    for name,pid in reversed(list(running.items())):
        if alive(name,pid):
            os.killpg(pid,signal.SIGTERM)
    deadline = time.monotonic()+15
    while time.monotonic()<deadline and any(alive(n,p) for n,p in running.items()):
        time.sleep(.2)
    for name,pid in running.items():
        if alive(name,pid):
            os.killpg(pid,signal.SIGKILL)
    (BASE/'pids.json').unlink(missing_ok=True)
    print('Local XMage server and proxy stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('install','start','status','stop'))
    args = parser.parse_args()
    globals()[args.command]()
