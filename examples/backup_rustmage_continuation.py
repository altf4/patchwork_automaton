#!/usr/bin/env python3
"""Finite-study backup emitter. Tar.gz only on stdout; never runs ML or games."""
import argparse,hashlib,json,sys
from pathlib import Path
from examples import backup_rustmage_pooled_study as base

base.MAX_JSON_BYTES=512*1024**2
base.MAX_TOTAL_BYTES=2*1024**3


def capture(root,after):
    root=Path(root).resolve()
    for attempt in range(3):
        members,manifest=base.capture_study(root,after_round=after,include_logs=True,max_log_bytes=256*1024)
        members.pop('backup-manifest.json',None)
        # Migration, terminal receipts and head-to-head results are top-level
        # regular JSON only. Never recurse into raw rollouts or other studies.
        for path in sorted(root.glob('*.json')):
            if path.name not in members:
                payload=base._read(path,base.MAX_JSON_BYTES);base._json(payload)
                members[path.name]=payload
        if base._read(root/'training/checkpoints/latest.json',base.MAX_JSON_BYTES)!=members['training/checkpoints/latest.json']:
            if attempt==2:raise RuntimeError('Checkpoint advanced during extended backup capture')
            continue
        if sum(map(len,members.values()))>base.MAX_TOTAL_BYTES:raise ValueError('Backup exceeds2GiB payload cap')
        manifest['members']=[dict(path=base._safe_name(name),size=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in sorted(members.items())]
        manifest['emitter_json_limit_bytes']=base.MAX_JSON_BYTES
        manifest['emitter_total_limit_bytes']=base.MAX_TOTAL_BYTES
        members['backup-manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        return members,manifest
    raise AssertionError('Unreachable capture loop')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--after-round',type=int,default=-1)
    a=p.parse_args(argv);members,_=capture(a.study_root,a.after_round);base.write_backup(members,sys.stdout.buffer)

if __name__=='__main__':main()
