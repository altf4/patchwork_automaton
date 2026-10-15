"""Evaluation-only turns-to-win metrics, independent of model and RL rewards."""
import math
import statistics
import hashlib
import json
from collections import Counter


def decision_state_key(state,kind,options,prompt):
    """Stable visible-state key for this Burn diagnostic, excluding revisions."""
    state={k:v for k,v in state.items() if k!='revision'}
    state['stack_order']=[row['id'] for row in state.get('objects',()) if row.get('zone')=='stack']
    for key in ('players','objects'):state[key]=sorted(state.get(key,()),key=lambda row:row['id'])
    payload=dict(state=state,kind=kind,options=sorted(options,key=lambda row:row['id']),prompt=prompt)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()


class RepeatGuard:
    """Bound repeated visible prompts; a benchmark cutoff, not a game loss."""
    def __init__(self,limit=25):
        if type(limit) is not int or limit<0:raise ValueError('Nonnegative repeat limit required')
        self.limit=limit;self.visits=Counter()

    def observe(self,key):
        if not self.limit:return False
        self.visits[key]+=1
        return self.visits[key]>self.limit


class OwnTurnCounter:
    def __init__(self,actor):
        self.actor=actor;self.turn_ids=set();self.first_active_player=None

    def observe(self,obs):
        # Many callbacks occur per turn, and opponent turns have their own
        # global numbers. Count the actor's distinct turns, including extra turns.
        if obs.actor==self.actor and obs.turn is not None and obs.phase is not None:
            if self.first_active_player is None and obs.active_player is not None:
                self.first_active_player=obs.active_player
            if obs.active_player==self.actor:self.turn_ids.add(obs.turn)

    @property
    def turns(self):return len(self.turn_ids)


def summarize(rows):
    wins=[r['own_turns_to_win'] for r in rows if r['status']=='win']
    counts={kind:sum(r['status']==kind for r in rows) for kind in ('win','loss','draw','cutoff','error')}
    ordered=sorted(wins)
    return dict(games=len(rows),**counts,win_rate=len(wins)/len(rows) if rows else None,
        failed_attempts=len(rows)-len(wins),failure_rate=(len(rows)-len(wins))/len(rows) if rows else None,
        mean_own_turns_to_win=statistics.mean(wins) if wins else None,
        median_own_turns_to_win=statistics.median(wins) if wins else None,
        p90_own_turns_to_win=ordered[math.ceil(.9*len(ordered))-1] if wins else None,
        fastest_win=min(wins) if wins else None,slowest_win=max(wins) if wins else None,
        wins_by_own_turn={str(t):sum(w<=t for w in wins)/len(rows) if rows else None for t in (4,5,6,10)},
        win_turn_histogram={str(t):wins.count(t) for t in sorted(set(wins))})
