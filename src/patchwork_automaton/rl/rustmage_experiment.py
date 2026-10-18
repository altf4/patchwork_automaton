"""Seeded mono-red mirrors and a separate inert-opponent competency metric."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import orjson

from ..backends.rustmage import RustMageMatch, domain_action
from ..baselines.foundations import foundations_deck, FoundationsAggro
from ..baselines.passive import passive_deck, PassivePolicy
from ..neural.experiment import TrajectoryWriter
from ..neural.rustmage_policy import RustMagePolicy
from ..neural.training import terminal_reward
from ..neural.rustmage_observation import learning_observation
from ..core import Handle
from .goldfish import OwnTurnCounter, summarize


def learning_view(match, viewer):
    """Read one seat's private response without generic immutable conversion."""
    return learning_observation(match, viewer)


def visible_state_fingerprint(raw):
    """Hash the complete private JSON state, excluding only monotonic revision.

    Sorting retains the original guard's independence from mapping insertion
    order. Native schema integers fit in 64 bits; no fields are projected away.
    """
    state = dict(raw)
    state.pop('revision', None)
    return hashlib.sha256(orjson.dumps(state, option=orjson.OPT_SORT_KEYS)).digest()


def play_game(*, seed, index, mode, model=None, batcher=None, version=0,
              output=None, max_decisions=6000, max_own_turns=60, repeat_limit=20,
              greedy=False, compact_views=True):
    """One worker owns one match; the inference service owns TensorFlow.

    Self-play uses terminal returns only. Cutoffs/errors have no reward and no
    training trajectories. Evaluation attempts are never replaced by retries.
    """
    async def play():
        started = time.monotonic()
        seat = index % 2
        start = f'P{(index//2)%2}'
        decks = [foundations_deck('red'), foundations_deck('red')]
        if mode != 'selfplay': decks[1-seat] = passive_deck()
        writers, policies, visits = {}, {}, Counter()
        row = dict(seed=seed, index=index, mode=mode, seat=seat, starting_player=start,
                   max_own_turns=max_own_turns, decisions=0, neural_choices=0,
                   status='error', own_turns_to_win=None, trajectories=[])
        try:
            for player in range(2):
                if mode != 'selfplay' and player != seat: policies[player] = PassivePolicy()
                elif mode == 'baseline': policies[player] = FoundationsAggro()
                else:
                    if output and mode == 'selfplay':
                        writers[player] = TrajectoryWriter(Path(output)/f'game-{index:05d}-p{player}.pkl.gz')
                    policies[player] = RustMagePolicy(model, batcher=batcher, version=version,
                        seed=seed*2+player, sink=writers.get(player), greedy=greedy)
            with RustMageMatch(*decks, seed=seed, starting_player=start) as match:
                handles = [match.games[f'player_{p}'] for p in range(2)]
                actors = [Handle('rustmage','player',f'P{p}') for p in range(2)]
                turns = [OwnTurnCounter(a) for a in actors]
                for step in range(max_decisions+1):
                    # Only each policy's own private view is ever given to it.
                    views = ([learning_view(match,f'P{p}') for p in range(2)] if compact_views
                             else [await g.snapshot() for g in handles])
                    for p, view in enumerate(views):
                        turns[p].observe(view)
                        if isinstance(policies[p], RustMagePolicy): policies[p].observe(view)
                    if views[0].terminal:
                        outcome = views[seat].outcome
                        row['status'] = 'draw' if outcome.draw else 'win' if outcome.won else 'loss'
                        row['winner'] = None if outcome.draw else seat if outcome.won else 1-seat
                        row['own_turns_to_win'] = turns[seat].turns if row['status'] == 'win' else None
                        row['terminal_rewards'] = [terminal_reward(v.outcome) for v in views]
                        for p, writer in writers.items():
                            if writer.count:
                                row['trajectories'].append(dict(path=str(writer.path), samples=writer.count,
                                    reward=terminal_reward(views[p].outcome)))
                        break
                    # A horizon is an evaluation/compute cutoff, never a fabricated
                    # loss or draw and never a turn-based training reward.
                    if step == max_decisions or (min(t.turns for t in turns) > max_own_turns
                            if mode == 'selfplay' else turns[seat].turns > max_own_turns):
                        row.update(status='cutoff', reason='decision_or_turn_limit')
                        break
                    p = next(p for p, v in enumerate(views) if v.decision is not None)
                    view = views[p]
                    # Full seat-private structured state, minus monotonic revision.
                    if repeat_limit:
                        # Hash the full private native state in both modes; the
                        # optimized feature projection must not weaken the guard.
                        key = visible_state_fingerprint(match._raw(view.actor.value))
                        visits[(p,key)] += 1
                        if visits[(p,key)] > repeat_limit:
                            row.update(status='cutoff', reason='repeated_visible_state'); break
                    policy = policies[p]
                    if isinstance(policy, RustMagePolicy):
                        action = policy.act(view)
                        row['neural_choices'] += bool(action.selections)
                        response = action.response
                    else: response = policy.choose(view)
                    # Trace compact decisions without hidden library order/hands.
                    option = next(o for o in view.decision.options if o.id == response.option)
                    chosen = (option.constraints['action'] if option.kind == 'finite'
                              else domain_action(option, response.value))
                    key = f"p{p}:{chosen['kind']}"
                    row.setdefault('actions', {})[key] = row.get('actions', {}).get(key, 0)+1
                    if compact_views:
                        # This worker exclusively owns the match. Native step/act
                        # still enforce actor, revision and full rules legality.
                        raw=match._raw(view.actor.value)
                        if option.kind=='finite':updated=match.native_game.step(raw,int(option.id.split(':')[1]))
                        else:updated=match.native_game.act(raw,domain_action(option,response.value))
                        match._after_action(updated, view.actor.value, raw['revision'])
                    else:
                        await handles[p].submit(view.decision.token, response)
                    row['decisions'] += 1
                row['own_turns_observed'] = [t.turns for t in turns]
                row['final_life'] = [p.life for p in (await handles[0].snapshot()).players]
        except Exception:
            row.update(status='error', error=traceback.format_exc(), trajectories=[])
        finally:
            for writer in writers.values(): writer.close()
            row['seconds'] = time.monotonic()-started
        return row
    return asyncio.run(play())


async def run_games(*, games, workers, seed_base, index_base=0, progress=None, **options):
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rustmage') as executor:
        async def one(i):
            async with semaphore:
                row = await loop.run_in_executor(executor, lambda: play_game(
                    seed=seed_base+i, index=index_base+i, **options))
                if progress: progress(row)
                return row
        return await asyncio.gather(*(one(i) for i in range(games)))


def goldfish_summary(rows, horizon):
    result = summarize(rows)
    # Prevent wins-only averages rewarding a policy that rarely finishes.
    # This is a descriptive, capped evaluation score, NOT an RL return.
    result['turn_horizon'] = horizon
    result['capped_turns_mean'] = statistics.mean(
        r['own_turns_to_win'] if r['status'] == 'win' else horizon+1 for r in rows) if rows else None
    result['capped_turns_failure_value'] = horizon+1
    return result


def paired_improvement(before, after, horizon, *, seed=9901):
    if [(r['seed'],r['seat'],r['starting_player']) for r in before] != [
            (r['seed'],r['seat'],r['starting_player']) for r in after]:
        raise ValueError('Expected matching evaluation seeds, seats and starting players')
    def score(r): return r['own_turns_to_win'] if r['status']=='win' else horizon+1
    differences = np.asarray([score(a)-score(b) for a,b in zip(before,after)], np.float64)
    if not len(differences): raise ValueError('Empty comparison')
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(5000,len(differences))).mean(axis=1)
    return dict(capped_turn_reduction=float(differences.mean()),
                bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                improved_attempts=int((differences>0).sum()),
                worsened_attempts=int((differences<0).sum()), tied_attempts=int((differences==0).sum()))
