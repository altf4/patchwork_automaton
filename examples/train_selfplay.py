"""Small CPU self-play RL run followed by fixed Burn-baseline evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import argparse
import asyncio
import json
from pathlib import Path
import time
from argparse import Namespace
from patchwork_automaton.baselines.modern_burn import modern_burn_deck
from patchwork_automaton.neural.policy import untrained_policy
from patchwork_automaton.neural.training import RolloutPolicy, TerminalActorCritic, terminal_reward
from patchwork_automaton.rl import XMageSelfPlay, EpisodeEnd
from examples.burn_baseline import run as evaluate


async def run(args):
    args.directory.mkdir(parents=True, exist_ok=True)
    initial = untrained_policy(seed=args.seed)
    policies = {seat: RolloutPolicy(initial.model, initial.encoder, seed=args.seed + i)
                for i, seat in enumerate(('player_0', 'player_1'))}
    def observe(obs):
        # History keys are actor-local even though both receive the listener.
        for policy in policies.values(): policy.observe(obs)
    factory = XMageSelfPlay(modern_burn_deck(), timeout=60, max_decisions=args.limit,
                            client_options={'on_observation': observe})
    trainer = TerminalActorCritic(initial.model, learning_rate=args.learning_rate, batch_size=args.batch_size)
    report = {'seed': args.seed, 'device': 'CPU', 'reward': {'win': 1, 'loss': -1, 'draw': 0},
              'gamma': 1, 'recurrent_gradient_steps': 1, 'episodes': [],
              'learning_rate': args.learning_rate, 'batch_size': args.batch_size}
    report_path = args.directory / 'training.json'
    try:
        for episode in range(args.games):
            for policy in policies.values(): policy.reset()
            runner = await factory.reset()
            start = time.monotonic()
            while True:
                turn = await runner.next()
                if isinstance(turn, EpisodeEnd): break
                action = await asyncio.to_thread(policies[turn.actor].act, turn.observation)
                await runner.submit(action.response)
                if runner.steps % 100 == 0:
                    print(json.dumps({'training_game': episode, 'decisions': runner.steps}), flush=True)
            summary = dict(episode=episode, decisions=runner.steps, truncated=turn.truncated,
                           reason=turn.reason, seconds=round(time.monotonic()-start, 3))
            if not turn.truncated and len(turn.observations) == 2:
                rewards = {seat: terminal_reward(obs.outcome) for seat, obs in turn.observations.items()}
                if sum(rewards.values()) != 0: raise ValueError('Inconsistent zero-sum results')
                if not initial.model.built: raise ValueError('No decisions collected')
                if not int(trainer.optimizer.iterations): initial.save(args.directory / 'initial.keras')
                summary['rewards'] = rewards
                summary['update'] = await asyncio.to_thread(trainer.update,
                    [(policies[seat].trajectory, reward) for seat, reward in rewards.items()])
                initial.save(args.directory / 'trained.keras')
            else:
                summary['update'] = None  # Never assign a loss to a truncated game.
            report['episodes'].append(summary)
            report['optimizer_steps'] = int(trainer.optimizer.iterations)
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(summary), flush=True)
    finally:
        await factory.close()
    if not int(trainer.optimizer.iterations): raise RuntimeError('No completed training episodes')
    evaluation = await evaluate(Namespace(opponent='neural', episodes=args.evaluation_games,
        seed=args.seed, skip_land_every=4, limit=args.limit,
        trace=args.directory / 'evaluation.jsonl', output=args.directory / 'evaluation.json',
        load=args.directory / 'trained.keras'))
    report['evaluation'] = {k: v for k, v in evaluation.items() if k != 'episodes'}
    report_path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games', type=int, default=12)
    p.add_argument('--evaluation-games', type=int, default=20)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--limit', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=0.0003)
    p.add_argument('--directory', type=Path, default=Path('.local/training/cpu-first'))
    args = p.parse_args()
    if min(args.games, args.evaluation_games, args.limit, args.batch_size) < 1: p.error('Counts must be positive')
    asyncio.run(run(args))
