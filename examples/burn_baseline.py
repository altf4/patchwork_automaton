"""Play/evaluate the deterministic weak Modern Burn baseline in local XMage."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from patchwork_automaton.baselines.modern_burn import BurnBaseline, modern_burn_deck, DECK_METADATA
from patchwork_automaton.rl import XMageSelfPlay, EpisodeEnd


async def run(args):
    baseline = BurnBaseline(skip_land_every=args.skip_land_every)
    opponent = BurnBaseline(skip_land_every=args.skip_land_every)
    neural = None
    if args.opponent == 'neural':
        from patchwork_automaton.neural.policy import NeuralPolicy, untrained_policy
        neural = (NeuralPolicy.load(args.load, seed=args.seed) if getattr(args, 'load', None)
                  else untrained_policy(seed=args.seed))
    factory = XMageSelfPlay(modern_burn_deck(), timeout=60, max_decisions=args.limit,
        client_options={'on_observation': neural.observe} if neural else {})
    results = []
    trace = Path(args.trace).open('w') if args.trace else None
    failures = []
    try:
        for episode in range(args.episodes):
            baseline.reset(); opponent.reset()
            if neural: neural.reset()
            baseline_seat = 'player_' + str(episode % 2)
            runner = await factory.reset()
            start = time.monotonic()
            counts, decisions = Counter(), Counter()
            while True:
                turn = await runner.next()
                if isinstance(turn, EpisodeEnd):
                    results.append(dict(episode=episode, baseline_seat=baseline_seat,
                        decisions=runner.steps, seconds=round(time.monotonic() - start, 3),
                        truncated=turn.truncated, reason=turn.reason, callbacks=dict(counts),
                        actor_decisions=dict(decisions),
                        outcomes={a: {'won': o.outcome.won, 'draw': o.outcome.draw} if o.outcome else None
                                  for a, o in turn.observations.items()}))
                    print(json.dumps(results[-1]), flush=True)
                    break
                obs = turn.observation
                try:
                    if turn.actor == baseline_seat:
                        response = baseline.choose(obs)
                    elif neural:
                        response = (await asyncio.to_thread(neural.act, obs)).response
                    else:
                        response = opponent.choose(obs)
                except Exception as error:
                    failures.append({'episode': episode, 'step': runner.steps, 'kind': obs.decision.kind if obs.decision else None,
                                     'error': str(error), 'native': thaw(obs.decision.native) if obs.decision else None,
                                     'view': thaw(obs.native), 'terminal': obs.terminal})
                    raise
                option = next(o for o in obs.decision.options if o.id == response.option)
                counts[obs.decision.kind] += 1; decisions[turn.actor] += 1
                if trace:
                    trace.write(json.dumps(dict(episode=episode, step=runner.steps, actor=turn.actor,
                        controller='baseline' if turn.actor == baseline_seat or not neural else 'neural',
                        turn=obs.turn, phase=obs.phase, kind=obs.decision.kind, label=option.label,
                        response={'option': response.option, 'value': response.value})) + '\n')
                    trace.flush()
                try:
                    await runner.submit(response)
                except Exception as error:
                    failures.append({'episode': episode, 'step': runner.steps, 'kind': obs.decision.kind,
                                     'error': str(error), 'native': thaw(obs.decision.native), 'view': thaw(obs.native)})
                    raise
                if runner.steps % 100 == 0:
                    print(json.dumps({'progress': runner.steps, 'episode': episode, 'turn': obs.turn}), flush=True)
    finally:
        await factory.close()
        if trace: trace.close()
        wins = sum(e['outcomes'].get(e['baseline_seat'], {}).get('won') is True for e in results
                   if e['outcomes'].get(e['baseline_seat']))
        output = dict(source=DECK_METADATA['source'], opponent=args.opponent, seed=args.seed,
                      skip_land_every=args.skip_land_every, episodes=results, failures=failures,
                      baseline_wins=wins, completed=len(results), optimizer_steps=0,
                      checkpoint=str(args.load) if getattr(args, 'load', None) else None)
        if args.output: Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    if any(r['truncated'] or len(r['outcomes']) != 2 or any(v is None for v in r['outcomes'].values()) for r in results):
        raise RuntimeError('Evaluation ended without both natural results')
    return output


def thaw(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping): return {str(k): thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [thaw(v) for v in value]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponent', choices=('baseline', 'neural'), default='baseline')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--skip-land-every', type=int, default=4)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--load', type=Path, help='Evaluate a saved neural checkpoint')
    args = parser.parse_args()
    if args.episodes < 1: parser.error('Episodes must be positive')
    asyncio.run(run(args))
