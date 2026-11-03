"""Recovery preserves actual saved games/checkpoints and rejects changed inputs."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from examples import train_hartford_ppo as driver
import test_hartford_driver as fixtures

class MemoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HartfordDriverTests('test_exact_default_recipe_and_no_warm_start_options')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root/'run'
        self.output.mkdir()
        self.config = self.fixture.config
        self.cp = self.output/'checkpoints/round-0000'
        self.cp.mkdir(parents=True)
        for name in driver.CHECKPOINT_FILES:
            (self.cp/name).write_bytes(name.encode())
        driver.atomic_json(self.cp/'state.json',dict(round=0,config=self.config,optimizer_steps=0,critic_optimizer_steps=0))
        self.checkpoint = driver.checkpoint_hashes(self.cp)
        self.original = self.root/'old-project'
        (self.original/'examples').mkdir(parents=True)
        (self.original/'src').mkdir()
        (self.original/'scripts').mkdir()
        (self.original/'examples/train_hartford_ppo.py').write_text('original frozen driver')
        (self.original/'scripts/build-direct-standard.py').write_text('original build')
        lib = self.original/'engine.so';lib.write_bytes(b'engine')
        self.readiness=dict(ready=True,engine=dict(library=str(lib),checkout=str(self.original),
            build_manifest='original.json',library_sha256=driver.sha256(lib),revision='unchanged'))
        self.old=dict(config=self.config,pins=dict(source=driver.source_manifest(self.original),readiness=self.readiness))
        self.pins=deepcopy(self.old['pins'])
        self.pins['source']['examples/train_hartford_ppo.py']='fixed driver hash'
        self.pins['readiness']['engine']['build_manifest']='relocated.json'
        self.report=dict(config=self.config,rounds=[],evaluations={},checkpoints={'0':self.checkpoint},completed_games=0)
        driver.atomic_json(self.output/'manifest.json',self.old)
        driver.atomic_json(self.output/'report.json',self.report)
        self.guard=self.root/'old-guard.json'
        driver.atomic_json(self.guard,dict(reason='low_memory',status='stopped',guard_exit_code=75,child_pid=2147483647))
        planned=driver.training_schedule(tuple(driver.HARTFORD_DECKS),round_id=1,seed=self.config['seed'])
        policy_hash=driver.sha256(self.cp/'policy.keras')
        self.planned=[dict(g,opponent_checkpoint_sha256=policy_hash) for g in planned]
        self.directory=self.output/'rollouts/round-0001'
        self.recorder=fixtures.Recorder()
        async def pin(number): pass
        asyncio.run(driver.collect_round(planned=self.planned,number=1,directory=self.directory,
            collectors=[self.recorder],pin_opponent=pin,progress=lambda *a,**kw:None))

    def adopt(self):
        driver.adopt_memory_fix(self.output,self.original,self.guard,self.config,self.pins)

    def test_adoption_preserves_checkpoint_and_reuses_all_games(self):
        before={str(p):driver.sha256(p) for p in self.directory.rglob('*') if p.is_file()}
        self.adopt()
        self.assertEqual(driver.checkpoint_hashes(self.cp),self.checkpoint)
        self.assertEqual(json.loads((self.output/'memory-recovery/manifest-before.json').read_text()),self.old)
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['pins'],self.pins)
        self.assertEqual(before,{str(p):driver.sha256(p) for p in self.directory.rglob('*') if p.is_file()})
        async def forbidden(number): raise AssertionError('Complete rollout must not start inference or play games')
        rows=asyncio.run(driver.collect_round(planned=self.planned,number=1,directory=self.directory,
            collectors=[],pin_opponent=forbidden,progress=lambda *a,**kw:None))
        self.assertEqual(len(rows),1000)
        self.assertEqual(sum(g['partition']=='training' for g in rows),800)
        self.assertEqual(sum(g['partition']=='validation' for g in rows),200)

    def test_changed_trajectory_fails_without_manifest_replacement(self):
        row=json.loads(next((self.directory/'results').glob('*.json')).read_text())
        Path(row['trajectories'][0]['path']).write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError,'trajectory changed'): self.adopt()
        self.assertEqual(json.loads((self.output/'manifest.json').read_text()),self.old)

    def test_incomplete_collection_fails(self):
        next((self.directory/'results').glob('*.json')).unlink()
        with self.assertRaisesRegex(ValueError,'1,000'): self.adopt()

    def test_unrelated_source_or_engine_changes_fail(self):
        self.pins['source']['src/other.py']='changed'
        with self.assertRaisesRegex(ValueError,'only the training driver'): self.adopt()
        self.pins['source'].pop('src/other.py')
        self.pins['readiness']['engine']['revision']='other'
        with self.assertRaisesRegex(ValueError,'Engine, card data'): self.adopt()

    def test_changed_checkpoint_or_committed_round_fails(self):
        (self.cp/'policy.keras').write_bytes(b'different weights')
        with self.assertRaisesRegex(ValueError,'Checkpoint zero changed'): self.adopt()

    def test_original_source_or_config_changes_fail(self):
        self.config['ppo_epochs']=3
        with self.assertRaisesRegex(ValueError,'unchanged checkpoint-zero'): self.adopt()

    def test_quiescence_joins_before_clearing_collectors(self):
        events=[]
        class Collector:
            async def close(self):
                await asyncio.sleep(0)
                events.append('closed')
        collectors=[Collector(),Collector()]
        asyncio.run(driver.quiesce_collectors(collectors))
        self.assertEqual(events,['closed','closed'])
        self.assertEqual(collectors,[])

if __name__=='__main__': unittest.main()
