"""Whole-game holdouts and read-only policy/value diagnostics.

Policy surrogate improvement is a local, noisy outcome-based proxy, not a win
rate or action accuracy. Recorded behavior probabilities and values stay fixed.
Validation never participates in optimization or advantage normalization.
"""
import numpy as np


def split_games(games, *, fraction=.2, seed=0, round_id=0):
    if not 0 <= fraction < 1:
        raise ValueError('Validation fraction must be in [0, 1)')
    ordered = sorted(games, key=lambda g: g['index'])
    if len({g['index'] for g in ordered}) != len(ordered):
        raise ValueError('Game indices must be unique')
    paths = [t['path'] for g in ordered for t in g.get('trajectories', ())]
    if len(paths) != len(set(paths)):
        raise ValueError('Trajectory paths must be unique across games and seats')
    if fraction and len(ordered) < 2:
        raise ValueError('A whole-game holdout needs at least two collected games')
    # Split attempts, before filtering outcomes. Never replace a validation
    # cutoff with a successful training game or assign fabricated returns.
    count = min(len(ordered)-1, max(1, int(np.ceil(len(ordered)*fraction)))) if fraction else 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 0x56414C]))
    selected = set(rng.permutation(len(ordered))[:count].tolist())
    train = [g for i,g in enumerate(ordered) if i not in selected]
    heldout = [g for i,g in enumerate(ordered) if i in selected]
    def counts(rows):
        usable = [g for g in rows if g.get('trajectories')]
        return dict(attempted_games=len(rows), terminal_games=len(usable),
                    samples=sum(t['samples'] for g in usable for t in g['trajectories']))
    return train, heldout, dict(method='whole_game_v1', fraction=fraction,
        seed=seed, round=round_id, training_indices=[g['index'] for g in train],
        validation_indices=[g['index'] for g in heldout],
        training=counts(train), validation=counts(heldout))


def training_normalization(games, scale_floor):
    """Exactly the PPO normalization, using training decisions exclusively."""
    from .experiment import read_trajectory
    selected = []
    for game in games:
        for trajectory in game.get('trajectories', ()):
            for sample in read_trajectory(trajectory['path']):
                if np.count_nonzero(sample.arrays['candidate_mask']) > 1:
                    selected.append(trajectory['reward']-sample.value)
    values = np.asarray(selected, np.float32)
    if not len(values):
        return dict(center=0., scale=1., training_choices=0, source='training_partition_only')
    std = float(values.std())
    return dict(center=float(values.mean()) if std > 1e-8 else 0.,
                scale=max(std, scale_floor) if std > 1e-8 else 1.,
                training_choices=len(values), source='training_partition_only')


def bootstrap_mean(rows, numerator, denominator, *, seed=1701, resamples=4000):
    """Resample whole games, keeping seats and decisions in the same block."""
    if not rows or sum(r[denominator] for r in rows) == 0:
        return None
    pairs = np.asarray([[r[numerator], r[denominator]] for r in rows], np.float64)
    totals = pairs.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = pairs[rng.integers(len(rows), size=(resamples, len(rows)))].sum(axis=1)
    means = draws[:,0]/np.maximum(draws[:,1], 1)
    return dict(mean=float(totals[0]/totals[1]),
                game_bootstrap_95_interval=np.quantile(means,[.025,.975]).tolist())


class GameValidator:
    def __init__(self, model, *, batch_size=32, clip_ratio=.2):
        import tensorflow as tf
        from .batching import example_batch, input_signature
        if batch_size < 1 or not 0 < clip_ratio < 1:
            raise ValueError('Invalid validation batch size or clipping ratio')
        self.model, self.batch_size, self.clip_ratio = model, batch_size, clip_ratio
        @tf.function(input_signature=[input_signature(example_batch(model.memory_size, model.feature_spec))], autograph=False)
        def forward(inputs):
            output = model(inputs, training=False)
            return output['value'][:,0], tf.nn.log_softmax(output['logits']), output['memory_out']
        self.forward = forward

    def evaluate(self, games, normalization, *, replay=True):
        from .experiment import read_trajectory
        from .features import collate
        from .ppo import value_prediction_diagnostics
        if normalization['scale'] <= 0:
            raise ValueError('Positive normalization scale required')
        rows = {g['index']:dict(index=g['index'], seed=g['seed'], samples=0, choices=0,
            value_samples=0, value_squared_error=0., behavior_value_squared_error=0.,
            value_error_change=0., policy_surrogate_gain=0., return_surrogate_gain=0.,
            policy_kl=0., max_state_kl=0.) for g in games if g.get('trajectories')}
        trajectories = [(g['index'],t) for g in games for t in g.get('trajectories', ())]
        targets, before, after = [], [], []
        for start in range(0, len(trajectories), self.batch_size):
            descriptors = trajectories[start:start+self.batch_size]
            streams = []
            for _, trajectory in descriptors:
                if trajectory['reward'] not in (-1., 0., 1.):
                    raise ValueError('Validation requires terminal win/loss/draw returns')
                records = list(read_trajectory(trajectory['path']))
                if len(records) != trajectory['samples']:
                    raise ValueError('Incomplete validation trajectory')
                streams.append(records)
            memory = np.zeros((len(streams), self.model.memory_size), np.float32)
            for step in range(max(map(len, streams), default=0)):
                active = [i for i,s in enumerate(streams) if step<len(s)]
                records = [streams[i][step] for i in active]
                inputs = collate([s.arrays for s in records])
                inputs['memory_in'] = memory[active] if replay else np.stack([s.memory for s in records])
                inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in records], np.float32)
                values, logp, updated = [np.asarray(x) for x in self.forward(inputs)]
                if not all(np.isfinite(x).all() for x in (values,logp,updated)):
                    raise FloatingPointError('Nonfinite validation inference')
                memory[active] = updated
                for j,i in enumerate(active):
                    record = records[j]
                    old = np.asarray(record.log_probs, np.float64)
                    mask = record.arrays['candidate_mask']
                    if (old.shape != mask.shape or not np.isfinite(old).all()
                            or not np.isclose(np.exp(old).sum(),1.,atol=1e-5)
                            or not 0 <= record.action < len(mask) or not mask[record.action]):
                        raise ValueError('Invalid frozen behavior distribution or action')
                    gi, trajectory = descriptors[i]
                    row = rows[gi]; reward = trajectory['reward']
                    row['samples'] += 1
                    if record.advance:
                        error = float((values[j]-reward)**2)
                        old_error = float((record.value-reward)**2)
                        row['value_samples'] += 1
                        row['value_squared_error'] += error
                        row['behavior_value_squared_error'] += old_error
                        row['value_error_change'] += error-old_error
                        targets.append(reward); before.append(record.value); after.append(values[j])
                    if np.count_nonzero(mask) > 1:
                        new = logp[j,:len(old)].astype(np.float64)
                        ratio = float(np.exp(new[record.action]-old[record.action]))
                        advantage = (reward-record.value-normalization['center'])/normalization['scale']
                        clipped = np.clip(ratio,1-self.clip_ratio,1+self.clip_ratio)
                        gain = min(ratio*advantage,clipped*advantage)-advantage
                        kl = max(0., float(np.sum(np.exp(old)*(old-new))))
                        row['choices'] += 1
                        row['policy_surrogate_gain'] += gain
                        row['return_surrogate_gain'] += (ratio-1)*reward
                        row['policy_kl'] += kl
                        row['max_state_kl'] = max(row['max_state_kl'],kl)
        rows = list(rows.values())
        valid = np.ones(len(targets), bool)
        choices = sum(r['choices'] for r in rows)
        return dict(scope='Frozen recorded observations/actions; terminal targets only. Policy surrogate gains are not win rates.',
            recurrent_states='replayed from zero with current weights' if replay else 'frozen behavior states',
            attempted_games=len(games), terminal_games=len(rows), normalization=normalization,
            value_before=value_prediction_diagnostics(targets,before,valid),
            value_after=value_prediction_diagnostics(targets,after,valid),
            value_mse=bootstrap_mean(rows,'value_squared_error','value_samples'),
            value_mse_change=bootstrap_mean(rows,'value_error_change','value_samples'),
            policy_surrogate_gain=bootstrap_mean(rows,'policy_surrogate_gain','choices'),
            return_surrogate_gain=bootstrap_mean(rows,'return_surrogate_gain','choices'),
            mean_policy_kl=sum(r['policy_kl'] for r in rows)/max(choices,1),
            max_state_kl=max((r['max_state_kl'] for r in rows),default=0.), games=rows)
