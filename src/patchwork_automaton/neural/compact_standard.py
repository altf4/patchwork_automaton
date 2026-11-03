"""Sparse, factored current-state inputs for Standard and Hartford policies.

No learned card-name identities. Object traits/facts are stored once per
decision; ordered action roles point into that table. Only nonzero coordinates
of object, context and action features are stored. History and previous-action
features are deliberately excluded from the versioned stateless architecture.
"""
from dataclasses import dataclass
import numpy as np
from .standard_features import StandardEncoder

SPEC = StandardEncoder.feature_spec
KINDS = len(StandardEncoder.actions)
ROLE = 1 + SPEC['semantics'] + SPEC['entities']
LOCAL = SPEC['candidates'] - 3 * ROLE
VERSION = 'sparse-factored-standard-v1'


def observation(arrays):
    """Compact worker payload, retaining legal-action masks for PPO audits."""
    record = arrays.get('_compact')
    if record is None:
        record = encode(arrays)
    return dict(_compact=record,candidate_mask=np.ones(len(record.roles),bool))


def decision(arrays):
    return arrays['_compact'] if '_compact' in arrays else encode(arrays)


def sparse(matrix):
    matrix = np.asarray(matrix, np.float32)
    row, col = np.nonzero(matrix)
    return (np.r_[0, np.cumsum(np.bincount(row, minlength=len(matrix)))].astype(np.int32),
            col.astype(np.uint16), matrix[row, col].copy())


def dense(csr, width):
    splits, columns, values = csr
    result = np.zeros((len(splits)-1, width), np.float32)
    result[np.repeat(np.arange(len(result)), np.diff(splits)), columns] = values
    return result


@dataclass
class CompactDecision:
    nodes: tuple
    local: tuple
    context: tuple
    roles: np.ndarray
    edges: np.ndarray
    refs: np.ndarray
    ref_values: np.ndarray
    board_nodes: int
    action: int
    deck: int
    kind: int
    dense_bytes: int

    @property
    def nbytes(self):
        return sum(a.nbytes for group in (self.nodes, self.local, self.context,
            (self.roles, self.edges, self.refs, self.ref_values)) for a in group) + 32


def encode(arrays, *, action=0, deck=0):
    candidates = arrays['candidates']
    if candidates.shape[1] != SPEC['candidates'] or not arrays['candidate_mask'].all():
        raise ValueError('Expected unpadded legal Standard candidates')
    if not 0 <= action < len(candidates):
        raise ValueError('Expert action is outside the legal options')
    if arrays['semantics'].shape[1] != 1:
        raise ValueError('Expected the single static trait row used by Standard')
    # Sentinel 0 is the missing role. Preserve every observed object, including
    # identical copies: their relations and board multiplicities still matter.
    board = np.concatenate([np.ones((len(arrays['entities']),1), np.float32),
        arrays['semantics'][:,0], arrays['entities']], axis=1)
    nodes = [np.zeros(ROLE, np.float32), *board]
    lookup = {row.tobytes():i for i,row in enumerate(nodes)}
    roles = np.empty((len(candidates),3), np.int32)
    for a in range(len(candidates)):
        for r in range(3):
            row = candidates[a, KINDS+r*ROLE:KINDS+(r+1)*ROLE]
            key = row.tobytes()
            if key not in lookup:
                lookup[key] = len(nodes); nodes.append(row.copy())
            roles[a,r] = lookup[key]
    local = np.concatenate([candidates[:,:KINDS], candidates[:,KINDS+3*ROLE:]], axis=1)
    edge = np.stack(np.nonzero(arrays['relations']),axis=1).astype(np.int32)
    if len(edge): edge[:,:2] += 1
    ref = np.stack(np.nonzero(arrays['candidate_refs']),axis=1).astype(np.int32)
    values = arrays['candidate_refs'][tuple(ref.T)].copy()
    return CompactDecision(sparse(np.asarray(nodes)), sparse(local),
        sparse(arrays['context'][None]), roles, edge, ref, values,
        len(board), int(action), int(deck), int(candidates[action,:KINDS].argmax()),
        sum(v.nbytes for v in arrays.values()))


def restore(record):
    """Reconstruct current-state dense inputs for the matched legacy control.

    Both controls are stateless: history/previous action/memory start at zero.
    Current entities, traits, context, action roles, relations and candidate
    references round-trip exactly, with unchanged legal action ordering.
    """
    nodes = dense(record.nodes, ROLE)
    board = nodes[1:1+record.board_nodes]
    local = dense(record.local, LOCAL)
    candidates = np.concatenate([local[:,:KINDS],
        nodes[record.roles].reshape(len(record.roles),3*ROLE),local[:,KINDS:]],axis=1)
    n, a = len(board), len(candidates)
    relations = np.zeros((n,n,4),np.float32)
    if len(record.edges):
        edge = record.edges.copy(); edge[:,:2] -= 1
        relations[tuple(edge.T)] = 1.
    refs = np.zeros((a,n),np.float32)
    refs[tuple(record.refs.T)] = record.ref_values
    return dict(entities=board[:,-SPEC['entities']:],card_ids=np.zeros(n,np.int32),
        entity_mask=np.ones(n,bool),semantics=board[:,1:1+SPEC['semantics']][:,None],
        semantic_mask=np.ones((n,1),bool),relations=relations,
        context=dense(record.context,SPEC['context'])[0],candidates=candidates,
        candidate_refs=refs,candidate_mask=np.ones(a,bool),selection=np.zeros(SPEC['candidates'],np.float32),
        history=np.zeros((1,SPEC['history']),np.float32),history_cards=np.zeros(1,np.int32),
        history_mask=np.zeros(1,bool))


class PackedBuilder:
    """Intern while reading a rollout; never retain all decoded decisions.

    The budget counts encoded array payloads, not Python/allocator overhead.
    New rows own their arrays so a small unique row cannot retain a large
    decoded observation through a NumPy view's base pointer.
    """
    def __init__(self,max_bytes=None):
        self.max_bytes=max_bytes;self.bytes=0;self.count=0
        self.tables={name:dict(lookup={},indices=[],values=[],lengths=[],row_ids=[])
                     for name in ('nodes','local','context')}
        self.parts={name:[] for name in ('node_counts','action_counts','roles','board','edge_counts',
            'edges','ref_counts','refs','ref_values','action','deck','kind')}

    def append(self,record,*,action=None):
        for name,t in self.tables.items():
            splits,columns,weights=getattr(record,name)
            for start,end in zip(splits[:-1],splits[1:]):
                col=columns[start:end];val=weights[start:end]
                key=(col.tobytes(),val.tobytes());identity=t['lookup'].get(key)
                if identity is None:
                    identity=len(t['indices']);t['lookup'][key]=identity
                    t['indices'].append(col.copy());t['values'].append(val.copy());t['lengths'].append(end-start)
                    self.bytes+=8*len(col)+4
                t['row_ids'].append(identity);self.bytes+=4
        n=len(record.nodes[0])-1;a=len(record.roles);p=self.parts
        p['node_counts'].append(n);p['action_counts'].append(a)
        p['roles'].append(record.roles.copy())
        p['board'].append(np.r_[False,np.ones(record.board_nodes,bool),np.zeros(n-1-record.board_nodes,bool)])
        p['edge_counts'].append(len(record.edges));p['edges'].append(record.edges.copy())
        p['ref_counts'].append(len(record.refs));p['refs'].append(record.refs.copy());p['ref_values'].append(record.ref_values.copy())
        for name in ('action','deck','kind'):p[name].append(action if name=='action' and action is not None else getattr(record,name))
        self.bytes+=record.roles.nbytes+n+record.edges.nbytes+record.refs.nbytes+record.ref_values.nbytes+32
        self.count+=1
        if self.max_bytes is not None and self.bytes>self.max_bytes:
            raise ValueError('Compact rollout exceeds its declared input budget')

    def finish(self):
        if not self.count:raise ValueError('Cannot pack an empty compact rollout')
        out={};p=self.parts
        for name,counts in (('node','node_counts'),('action','action_counts'),('edge','edge_counts'),('ref','ref_counts')):
            out[name+'_splits']=np.r_[0,np.cumsum(p[counts])].astype(np.int32)
        for name,t in self.tables.items():
            out[name+'_splits']=np.r_[0,np.cumsum(t['lengths'])].astype(np.int32)
            out[name+'_indices']=np.concatenate(t['indices']).astype(np.int32)
            out[name+'_values']=np.concatenate(t['values'])
            out[name+'_row_ids']=np.asarray(t['row_ids'],np.int32)
        for name in ('roles','board','edges','refs','ref_values'):out[name]=np.concatenate(p[name])
        for name in ('action','deck','kind'):out[name]=np.asarray(p[name],np.int32)
        size=sum(v.nbytes for v in out.values())
        if self.max_bytes is not None and size>self.max_bytes:raise ValueError('Compact rollout exceeds its declared input budget')
        return out


def pack(records):
    builder=PackedBuilder()
    for record in records:builder.append(record)
    return builder.finish()
