"""Opt-in identity sidecar. Preserve v1 tensors and exact ordered object references."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from . import compact_standard as base
from .standard_features import StandardEncoder
from .standard_choice_features import _choice_objects,_with_visible_objects
from .rustmage_features import visible_identity
from .identity_features import CARDS,card_id,manifest as identity_manifest

VERSION='sparse-identity-standard-v2'
SPEC=base.SPEC

@dataclass
class IdentityDecision(base.CompactDecision):
    identity_ids: np.ndarray
    identity_base_nodes: np.ndarray
    identity_roles: np.ndarray
    identity_context: int
    input_format=VERSION

    @property
    def nbytes(self):
        return super().nbytes+self.identity_ids.nbytes+self.identity_base_nodes.nbytes+self.identity_roles.nbytes+4

def _normalized_packet(packet):
    if not packet:return {}
    source=packet.get('source') or {}
    hidden=any(p.get(k) for p in (packet,source,source.get('object') or {})
        for k in ('hideInfo','faceDown','face_down'))
    return dict(packet,card=(source.get('object') or {}).get('card',packet.get('card')),
        source_card=None if hidden else source.get('card',packet.get('source_card')))

class IdentityEncoder(StandardEncoder):
    def manifest(self):
        result=super().manifest();result.pop('fingerprint')
        result.update(representation=VERSION,identity=identity_manifest())
        result['fingerprint']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
        return result

    def encode(self,obs,*,choices=None):
        choices=choices if choices is not None else self.expand_options(obs.decision)
        obs=_with_visible_objects(obs,_choice_objects(obs,choices))
        result=super().encode(obs,choices=choices)
        result.arrays=self.enrich(obs,choices,result.arrays)
        return result

    def enrich(self,obs,choices,arrays):
        """Accept reference or parity-checked native v1 arrays from the same private view."""
        obs=_with_visible_objects(obs,_choice_objects(obs,choices))
        objects=list(obs.objects.values());players=sorted(obs.players,key=lambda p:p.id!=obs.actor)
        if len(arrays['entities'])!=len(players)+len(objects) or len(choices)!=len(arrays['candidates']):
            raise ValueError('Identity sidecar does not match native entity/action ordering')
        record=base.encode(arrays)
        ids=[0]*(1+len(players))+[card_id(visible_identity(o.properties)) for o in objects]
        nodes=list(range(len(ids)))
        lookup={p.id.value:i+1 for i,p in enumerate(players)}
        lookup.update({int(o.id.value):i+1+len(players) for i,o in enumerate(objects)})
        raw=obs.native;stack=[_normalized_packet(s) for s in list(raw['stack'])+list(raw['waiting_triggers'])]
        by_id={s['id']:s for s in stack}
        pending=_normalized_packet(raw.get('announcing') or raw.get('resolving'))
        aux={}
        def named(name,base_index):
            identity=card_id(name)
            if not identity:return 0
            key=(identity,int(base_index))
            if key not in aux:
                aux[key]=len(ids);ids.append(identity);nodes.append(int(base_index))
            return aux[key]
        def target(value,base_index):
            if not value:return 0
            if 'Player' in value:return lookup[value['Player']]
            if 'Card' in value:return lookup.get(value['Card'],0)
            if 'Spell' in value:return lookup.get(value['Spell'],0)
            if 'Stack' in value:
                packet=by_id.get(value['Stack'],{})
                index=lookup.get(packet.get('card'))
                return named(packet.get('source_card'),base_index) if index is None else index
            raise ValueError('Unknown identity target')
        roles=[]
        for i,(_,action) in enumerate(choices):
            r=self.action_roles(action)
            source=by_id.get(action.get('id'),{}) if action['kind']=='PutTrigger' else pending
            index=lookup.get(r.get('card',r.get('blocker',source.get('card'))))
            # A present but concealed object stays unknown; never fall back to
            # a name in another raw field for such an object.
            if index is None:index=named(source.get('source_card'),record.roles[i,0])
            roles.append([index,target(r.get('target'),record.roles[i,1]),lookup.get(r.get('attacker'),0)])
        context=lookup.get(pending.get('card'))
        if context is None:context=named(pending.get('source_card'),0)
        rich=IdentityDecision(**record.__dict__,identity_ids=np.asarray(ids,np.int32),
            identity_base_nodes=np.asarray(nodes,np.int32),identity_roles=np.asarray(roles,np.int32),
            identity_context=context)
        return dict(arrays,_compact=rich)

def observation(arrays):
    record=arrays.get('_compact')
    if not isinstance(record,IdentityDecision):raise ValueError('Identity model requires freshly enriched private observations')
    return dict(_compact=record,candidate_mask=np.ones(len(record.roles),bool))

def decision(arrays):return observation(arrays)['_compact']

class PackedBuilder(base.PackedBuilder):
    def __init__(self,max_bytes=None):
        super().__init__(max_bytes);self.identities=[]

    def append(self,record,*,action=None):
        if not isinstance(record,IdentityDecision):raise ValueError('Cannot adopt v1 rollouts into identity inputs')
        n=len(record.identity_ids)
        integer_arrays=(record.identity_ids,record.identity_base_nodes,record.identity_roles)
        if (any(a.dtype!=np.dtype('int32') for a in integer_arrays)
                or record.identity_ids.shape!=(n,) or n<1 or record.identity_ids[0]!=0
                or np.any(record.identity_ids<0) or np.any(record.identity_ids>len(CARDS))
                or record.identity_base_nodes.shape!=(n,) or record.identity_roles.shape!=(len(record.roles),3)
                or np.any(record.identity_base_nodes<0) or np.any(record.identity_base_nodes>=len(record.nodes[0])-1)
                or np.any(record.identity_roles<0) or np.any(record.identity_roles>=n)
                or not 0<=record.identity_context<n):raise ValueError('Invalid identity sidecar references')
        super().append(record,action=action)
        self.bytes+=record.nbytes-super(IdentityDecision,record).nbytes
        if self.max_bytes is not None and self.bytes>self.max_bytes:raise ValueError('Compact rollout exceeds its declared input budget')
        self.identities.append((record.identity_ids.copy(),record.identity_base_nodes.copy(),
            record.identity_roles.copy(),record.identity_context))

    def finish(self):
        out=super().finish();rs=self.identities
        splits=np.r_[0,np.cumsum([len(r[0]) for r in rs])].astype(np.int32)
        out['identity_splits']=splits
        out['identity_ids']=np.concatenate([r[0] for r in rs])
        out['identity_base_nodes']=np.concatenate([r[1]+out['node_splits'][i] for i,r in enumerate(rs)])
        out['identity_roles']=np.concatenate([r[2]+splits[i] for i,r in enumerate(rs)])
        out['identity_context']=np.asarray([r[3]+splits[i] for i,r in enumerate(rs)],np.int32)
        if self.max_bytes is not None and sum(a.nbytes for a in out.values())>self.max_bytes:
            raise ValueError('Compact rollout exceeds its declared input budget')
        return out

def pack(records):
    builder=PackedBuilder()
    for record in records:builder.append(record)
    return builder.finish()
