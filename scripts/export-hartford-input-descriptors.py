#!/usr/bin/env python3
"""Export experimental descriptors from evaluated typed rules in a guarded archive."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'src/patchwork_automaton/data/hartford-input-descriptors.json')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('traits_export',ROOT/'scripts/export-rustmage-card-traits.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.HELPER=ROOT/'scripts/hartford-input-descriptors.rs'
    folder,revision,helper_hash=helper.prepare(a.source,a.revision)
    # The pinned engine uses a catalogue/core workspace. Its rule types are
    # public; attach the visitor to the root crate instead of the legacy file.
    root_lib=folder/'src/lib.rs'
    text=root_lib.read_text()
    root_lib.write_text(text.replace('pub use cards::card_traits_export;', 'pub mod card_traits_export;'))
    command=[sys.executable,'scripts/resource_guard.py','--','cargo','build','--locked','--bin','export_card_traits']
    if a.prepare_only:
        print(json.dumps(dict(directory=str(folder),command=command)));return
    subprocess.run(command,cwd=folder,check=True,env=dict(os.environ,CARGO_INCREMENTAL='0',CARGO_PROFILE_DEV_DEBUG='0'))
    raw=subprocess.check_output([str(folder/'target/debug/export_card_traits')],cwd=folder)
    (folder/'hartford-input-descriptors-raw.json').write_bytes(raw)
    data=json.loads(raw)
    old=json.loads((ROOT/'src/patchwork_automaton/data/rustmage-card-traits.json').read_text())
    rows={c['index']:c for c in data['cards']}
    if set(rows)!={c['index'] for c in old['cards']}:raise ValueError('Catalogue identities differ from the saved feature catalogue')
    cards={}
    for c in old['cards']:
        row=rows[c['index']]
        if row['identity']!=c.get('serialization_identity',c['identity']):raise ValueError('Identity mismatch')
        cards[c['identity']]=dict(entry=row['entry'],spells=row['spells'])
    payload=dict(schema=1,cards=cards)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    data=dict(payload,manifest=dict(runtime_revision=revision,catalogue_version=data['catalogue_version'],
        feature_traits_sha256=old['manifest']['traits_sha256'],extractor_sha256=helper_hash,
        payload_sha256=digest,scope='Exact entry variants and narrow spell-counter descriptors; no runtime rules changes'))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps(data['manifest'],indent=2))

if __name__=='__main__':main()
