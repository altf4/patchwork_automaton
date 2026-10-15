"""Run an untrained Keras policy on both seats of local XMage games."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time

# Explicit CPU defaults for the smoke runner; users may override these.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from patchwork_automaton.core import CardSpec, DeckSpec
from patchwork_automaton.rl import XMageSelfPlay, EpisodeEnd
from patchwork_automaton.neural.policy import NeuralPolicy, untrained_policy


async def run(args):
    policy = (NeuralPolicy.load(args.load, seed=args.seed, temperature=args.temperature)
              if args.load else untrained_policy(seed=args.seed, temperature=args.temperature))
    cards = ((CardSpec('Mountain', 24), CardSpec('Lightning Bolt', 36)) if args.deck == 'burn'
             else (CardSpec('Mountain', 24), CardSpec('Lightning Bolt', 12),
                   CardSpec('Goblin Piker', 12), CardSpec('Raging Goblin', 12)))
    deck = DeckSpec(cards, 'Untrained neural ' + args.deck)
    factory = XMageSelfPlay(deck, deck_type='Constructed - Freeform Unlimited',
                           timeout=60, max_decisions=args.limit,
                           client_options={'on_observation': policy.observe})
    summaries = []
    trace = Path(args.trace).open('w') if args.trace else None
    try:
        for episode in range(args.episodes):
            policy.reset()
            runner = await factory.reset()
            start = time.monotonic()
            kinds, selected, selections = Counter(), Counter(), Counter()
            entropy_sum = 0.
            while True:
                turn = await runner.next()
                if isinstance(turn, EpisodeEnd):
                    summary = dict(episode=episode, decisions=runner.steps,
                        seconds=round(time.monotonic() - start, 3), truncated=turn.truncated,
                        reason=turn.reason, callbacks=dict(kinds), selected=dict(selected),
                        selections=dict(selections), mean_entropy=entropy_sum / max(runner.steps, 1),
                        final_life={seat: {('self' if p.id == obs.actor else 'opponent'): p.life for p in obs.players}
                                    for seat, obs in turn.observations.items()},
                        outcomes={seat: {'won': obs.outcome.won, 'draw': obs.outcome.draw}
                                  if obs.outcome else None for seat, obs in turn.observations.items()})
                    summaries.append(summary)
                    print(json.dumps(summary), flush=True)
                    break
                observation = turn.observation
                # Keep the gateway event loop responsive while TensorFlow runs.
                record = await asyncio.to_thread(policy.act, observation)
                option = next(o for o in observation.decision.options if o.id == record.response.option)
                kinds[observation.decision.kind] += 1
                selected[option.label] += 1
                selections[option.kind] += 1
                entropy_sum += record.selections[0].entropy
                if trace:
                    trace.write(json.dumps(dict(step=runner.steps, seat=turn.actor, turn=observation.turn,
                        phase=observation.phase, decision=observation.decision.kind, option=option.id,
                        label=option.label, value=record.response.value, estimate=record.value,
                        log_probability=record.log_probability,
                        choices=[vars(s) for s in record.selections])) + '\n')
                    trace.flush()
                await runner.submit(record.response)
                if runner.steps % 100 == 0:
                    print(json.dumps({'progress': runner.steps, 'episode': episode, 'turn': observation.turn}), flush=True)
        if args.save: policy.save(args.save)
    finally:
        await factory.close()
        if trace: trace.close()
        if args.output:
            Path(args.output).write_text(json.dumps({'seed': args.seed, 'deck': args.deck,
                'initialization': 'loaded' if args.load else 'random', 'optimizer_steps': 0,
                'model_parameters': policy.model.count_params() if policy.model.built else None,
                'episodes': summaries}, indent=2) + '\n')
    if any(s['truncated'] or len(s['outcomes']) != 2 or any(v is None for v in s['outcomes'].values()) for s in summaries):
        raise RuntimeError('A live episode did not finish with both server results')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--limit', type=int, default=5000, help='Decision limit truncates; it never forces a win by concession')
    parser.add_argument('--seed', type=int, default=17, help='Model/action RNG seed; XMage shuffles are not seeded')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--deck', choices=('burn', 'combat'), default='burn')
    parser.add_argument('--load', type=Path)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    if arguments.episodes < 1 or arguments.limit < 1: parser.error('Episodes and limit must be positive')
    asyncio.run(run(arguments))
