"""Native decisions, learned action bindings, metrics and checkpoint isolation."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_neural import AVAILABLE
from patchwork_automaton.core import Decision, Option
from patchwork_automaton.errors import UnsupportedInteraction

if AVAILABLE:
    import keras
    import numpy as np
    from patchwork_automaton.backends.rustmage import RustMageMatch, action_options, _plain
    from patchwork_automaton.baselines.foundations import foundations_deck, FoundationsAggro
    from patchwork_automaton.neural.batching import example_batch, InferenceBatcher
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.model import PolicyNetwork
    from patchwork_automaton.neural.rustmage_features import RustMageEncoder, expand_options, ACTIONS
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.neural.experiment import save_checkpoint, load_checkpoint, make_trainer
    from patchwork_automaton.rl.rustmage_experiment import play_game, run_games, goldfish_summary, paired_improvement, learning_view
    from patchwork_automaton.neural.history import History

NATIVE = (Path(os.environ.get('RUSTMAGE_ROOT','.local/rustmage'))/'target/release/librustmage.so').is_file()


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class RustDomainMetricTests(unittest.TestCase):
    def test_repeat_guard_ignores_only_revision_and_mapping_order(self):
        from copy import deepcopy
        from patchwork_automaton.rl.rustmage_experiment import visible_state_fingerprint
        raw = dict(revision=1, viewer='P0', hand=[dict(id=2**64-1, name='Éowyn', tapped=False)],
                   actions=[dict(kind='Pass')], phase=None)
        equivalent = dict(reversed(list(raw.items())))
        equivalent['revision'] = 900
        equivalent['hand'] = [dict(reversed(list(raw['hand'][0].items())))]
        self.assertEqual(visible_state_fingerprint(raw), visible_state_fingerprint(equivalent))
        for key, value in [('viewer', 'P1'), ('actions', []), ('phase', 'Main1'),
                           ('hand', [dict(raw['hand'][0], tapped=True)])]:
            changed = deepcopy(raw); changed[key] = value
            self.assertNotEqual(visible_state_fingerprint(raw), visible_state_fingerprint(changed))

    def test_unbatched_and_legacy_inference_requests_pack_identically(self):
        from patchwork_automaton.neural.batching import Request, pack_requests
        first = example_batch(8)
        second = {k: v.copy() for k, v in first.items()}
        for key in ('candidates', 'candidate_refs', 'candidate_mask'):
            second[key] = np.repeat(second[key], 3, axis=1)
        requests = [Request(first, 0, (), None, 0), Request(second, 0, (), None, 0)]
        _, expected = pack_requests(requests)
        requests[1] = Request({k: v[0] for k, v in second.items()}, 0, (), None, 0, True)
        _, actual = pack_requests(requests)
        for key, value in expected.items(): np.testing.assert_array_equal(value, actual[key])

    def test_exact_payment_domain_and_deduplication(self):
        raw=dict(actions=[dict(kind='PayCosts',payment=[1,0,0,1,0,0])],
            mana_payment=dict(required=[0,0,0,1,0,0],available=[2,0,0,3,0,0],generic=1))
        decision=Decision(None,'pay_costs','',action_options(raw))
        choices=expand_options(decision)
        self.assertEqual({tuple(a['payment']) for _,a in choices},{(1,0,0,1,0,0),(0,0,0,2,0,0)})
        self.assertEqual(len(choices),2)
        enormous=Decision(None,'damage','',action_options(dict(actions=[],damage_assignment=
            dict(target={'Player':'P1'},minimum=0,maximum=2**80))))
        with self.assertRaises(UnsupportedInteraction):expand_options(enormous)
        sparse=Decision(None,'mana_lot_spend','',action_options(dict(actions=[],mana_lot_spend=
            dict(minimum=0,maximum=100,values=[0,100]))))
        self.assertEqual([a['amount'] for _,a in expand_options(sparse)],[0,100])

    def test_concede_setting_explicit_and_no_other_pruning(self):
        d=Decision(None,'priority','',action_options(dict(actions=[{'kind':'Concede'},{'kind':'Pass'},
            {'kind':'AnnounceActivate','card':1,'ability':0,'counter_payment':{}}])))
        self.assertEqual(len(expand_options(d)),2)
        self.assertEqual(len(expand_options(d,allow_concede=True)),3)
        self.assertNotEqual(RustMageEncoder().manifest()['fingerprint'],
                            RustMageEncoder(allow_concede=True).manifest()['fingerprint'])

    def test_failures_never_improve_the_wins_only_metric_silently(self):
        def row(i,status,turn=None):return dict(seed=i,seat=i%2,starting_player='P0',status=status,own_turns_to_win=turn)
        before=[row(0,'win',8),row(1,'cutoff'),row(2,'loss'),row(3,'error')]
        after=[row(0,'win',5),row(1,'win',7),row(2,'win',8),row(3,'draw')]
        summary=goldfish_summary(before,10)
        self.assertEqual(summary['mean_own_turns_to_win'],8)
        self.assertEqual(summary['capped_turns_mean'],10.25)
        self.assertEqual(summary['failed_attempts'],3)
        self.assertEqual(paired_improvement(before,after,10)['improved_attempts'],3)
        with self.assertRaises(ValueError):paired_improvement(before,after[::-1],10)


@unittest.skipUnless(AVAILABLE and NATIVE, 'Requires neural extra and built RustMage')
class RustNeuralTests(unittest.TestCase):
    def initial(self,seed=19):
        match=RustMageMatch(foundations_deck('red'),seed=seed)
        self.addCleanup(match.close)
        return match,asyncio.run(match.games['player_0'].snapshot())

    def model(self):
        encoder=RustMageEncoder()
        model=PolicyNetwork(encoder.card_count,width=16,memory_size=8,blocks=0,
                            feature_spec=encoder.feature_spec)
        model(example_batch(8,encoder.feature_spec))
        return model

    def test_private_numeric_inputs_and_text_independence(self):
        match,obs=self.initial()
        encoder=RustMageEncoder();first=encoder.encode(obs)
        self.assertEqual(len(obs.objects),7)
        self.assertTrue(all(o.controller==obs.actor for o in obs.objects.values()))
        changed=replace(obs,decision=replace(obs.decision,message='display only',
            options=tuple(replace(o,label='arbitrary text') for o in obs.decision.options)),
            players=tuple(replace(p,name='display only') for p in obs.players))
        for key,value in first.arrays.items():
            self.assertIn(value.dtype.kind,'bif')
            np.testing.assert_array_equal(value,encoder.encode(changed).arrays[key])
        with self.assertRaises(UnsupportedInteraction):encoder.encode(replace(obs,backend='xmage'))

    def test_source_target_and_card_identity_are_distinct(self):
        _,obs=self.initial()
        cards=list(obs.objects)
        actions=[{'kind':'ChooseTarget','target':{'Player':p.id.value}} for p in obs.players]
        actions += [{'kind':'AnnounceCast','card':int(h.value)} for h in cards]
        actions += [dict(kind='Block',blocker=int(cards[0].value),attacker=int(cards[1].value)),
                    dict(kind='Block',blocker=int(cards[1].value),attacker=int(cards[0].value))]
        fake=replace(obs,decision=replace(obs.decision,options=action_options(dict(actions=actions))))
        arrays=RustMageEncoder().encode(fake).arrays
        self.assertFalse(np.array_equal(arrays['candidates'][0],arrays['candidates'][1]))
        # Direction changes source/other role rows even with the same pooled refs.
        if obs.objects[cards[0]].name != obs.objects[cards[1]].name:
            self.assertFalse(np.array_equal(arrays['candidates'][-1],arrays['candidates'][-2]))
        self.assertEqual(arrays['candidate_refs'][0,0],1)
        self.assertEqual(arrays['candidate_refs'][1,1],1)

    def test_full_native_game_encodes_payments_targets_and_triggers(self):
        async def run():
            with RustMageMatch(foundations_deck('red'),seed=45) as match:
                kinds=set();encoder=RustMageEncoder();policy=FoundationsAggro()
                for _ in range(2000):
                    raw=match._raw('P0')
                    if raw['outcome']:return kinds
                    game=match.games['player_'+raw['actor'][-1]]
                    obs=await game.snapshot();encoded=encoder.encode(obs)
                    chosen=policy.choose(obs)
                    self.assertIn(chosen,encoded.responses)
                    kinds.add(obs.decision.kind)
                    await game.submit(obs.decision.token,chosen)
                self.fail('Game did not finish')
        kinds=asyncio.run(run())
        self.assertTrue({'priority','pay_costs','targets','order_triggers'}<=kinds)

    def test_compact_views_preserve_every_tensor_and_private_history(self):
        async def run():
            encoder=RustMageEncoder()
            full_history=History(encoder.registry);compact_history=History(encoder.registry)
            with RustMageMatch(foundations_deck('red'),seed=45) as match:
                for _ in range(2000):
                    full=[await g.snapshot() for g in match.games.values()]
                    compact=[learning_view(match,f'P{p}') for p in range(2)]
                    for left,right in zip(full,compact):
                        self.assertEqual(left.outcome,right.outcome)
                        self.assertEqual(left.decision,right.decision)
                        full_history.observe(left);compact_history.observe(right)
                        a_history,b_history=full_history.take(left),compact_history.take(right)
                        for key,value in a_history.items():
                            np.testing.assert_array_equal(value,b_history[key])
                        if left.decision is not None:
                            a,b=encoder.encode(left),encoder.encode(right)
                            self.assertEqual(a.responses,b.responses)
                            for key,value in a.arrays.items():np.testing.assert_array_equal(value,b.arrays[key])
                    if full[0].terminal:return
                    obs=next(v for v in full if v.decision)
                    await match.games['player_'+obs.actor.value[-1]].submit(obs.decision.token,FoundationsAggro().choose(obs))
                self.fail('Game did not finish')
        asyncio.run(run())

    def test_compact_and_full_baseline_games_replay_identically(self):
        full=play_game(seed=45,index=0,mode='baseline',compact_views=False)
        compact=play_game(seed=45,index=0,mode='baseline',compact_views=True)
        for key in ('status','own_turns_to_win','actions','decisions','terminal_rewards','final_life'):
            self.assertEqual(full[key],compact[key])

    def test_round_trip_checkpoint_optimizer_and_schema_rejection(self):
        model=self.model()
        config=dict(algorithm='ppo',learning_rate=.0001,train_batch=2,ppo_epochs=1,seed=19)
        trainer=make_trainer(model,config);trainer.compile()
        _,obs=self.initial()
        arrays=RustMageEncoder().encode(obs).arrays
        batch=collate([arrays]);batch.update(memory_in=np.zeros((1,8),np.float32),advance_memory=np.ones((1,1),np.float32))
        before=np.asarray(model(batch)['logits'])
        with tempfile.TemporaryDirectory() as directory:
            root=save_checkpoint(directory,model,trainer,dict(round=0,config=config),policy_type=RustMagePolicy)
            loaded,restored,state=load_checkpoint(directory)
            np.testing.assert_allclose(before,np.asarray(loaded(batch)['logits']),atol=1e-6)
            self.assertEqual(trainer.rng.bit_generator.state,restored.rng.bit_generator.state)
            for left,right in zip(trainer.optimizer.variables,restored.optimizer.variables):
                np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
            with self.assertRaises(ValueError):RustMagePolicy.load(root/'policy.keras',encoder=RustMageEncoder(allow_concede=True))
            path=root/'policy.manifest.json';manifest=json.loads(path.read_text())
            reordered={**manifest,'actions':dict(reversed(list(manifest['actions'].items())))}
            path.write_text(json.dumps(reordered))
            with self.assertRaises(ValueError):RustMagePolicy.load(root/'policy.keras')
            manifest['backend']='xmage';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):RustMagePolicy.load(root/'policy.keras')

    def test_passive_opponent_and_cutoffs_have_no_training_returns(self):
        result=play_game(seed=45,index=0,mode='baseline',max_own_turns=20)
        self.assertEqual(result['status'],'win')
        self.assertGreater(result['own_turns_to_win'],0)
        self.assertFalse(any(result['actions'].get('p1:'+k,0) for k in ('PlayLand','AnnounceCast','Attack','Concede')))
        cutoff=play_game(seed=45,index=0,mode='baseline',max_decisions=1)
        self.assertEqual(cutoff['status'],'cutoff')
        self.assertEqual(cutoff['trajectories'],[])
        self.assertNotIn('terminal_rewards',cutoff)

    def test_batched_model_runs_two_private_games(self):
        async def run():
            model=self.model();batcher=await InferenceBatcher(model,max_batch=2,wait_ms=2).start()
            try:
                rows=await run_games(games=2,workers=2,seed_base=31,mode='goldfish',
                    model=model,batcher=batcher,max_decisions=150,max_own_turns=3)
                self.assertTrue(all(r['status']!='error' for r in rows),rows)
                self.assertGreater(batcher.stats()['requests'],0)
                self.assertEqual(batcher.stats()['graph_traces'],1)
            finally:await batcher.close()
        asyncio.run(run())
