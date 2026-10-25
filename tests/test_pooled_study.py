"""Finite-budget and owned-process cleanup checks without games or learning."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples import run_rustmage_pooled_study as study


class PooledStudyTests(unittest.TestCase):
    def test_finite_budget_rejects_extra_training(self):
        with self.assertRaisesRegex(ValueError,'Finite study budget'):
            study.run({'target_games':80000,'milestones':list(study.MILESTONES)})

    def test_command_keeps_learning_recipe_and_resumability(self):
        command=study.train_command('python','out',dict(train_batch=512,ppo_epochs=2,
            pooled_collector_config='/fixed/pool.json',algorithm='ppo',concurrent_updates=True,
            guard_audit=False,advantage_scale_floor=1.),40)
        self.assertIn('--resume',command)
        self.assertEqual(command[command.index('--train-batch')+1],'512')
        self.assertEqual(command[command.index('--rounds')+1],'40')
        self.assertNotIn('--algorithm',command)
        self.assertNotIn('--guard-audit',command)

    def test_native_heartbeat_binds_exact_owner(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.json'
            value=dict(utc=datetime.now(timezone.utc).isoformat(),manifest_sha256='a'*64,
                       windows_pid=10,creation_utc='immutable-creation')
            p.write_text(json.dumps(value))
            m=dict(native_liveness=str(p),native_manifest_sha256='a'*64)
            self.assertTrue(study.native_alive(m))
            value['windows_pid']=11;p.write_text(json.dumps(value))
            self.assertFalse(study.native_alive(m))
            value['windows_pid']=10;value['utc']='2000-01-01T00:00:00+00:00';p.write_text(json.dumps(value))
            self.assertFalse(study.native_alive(m))

    def test_exited_leader_cannot_hide_an_ignoring_worker(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'worker.pid';log=Path(d)/'child.log'
            worker="import os,signal,time,pathlib; signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("+repr(str(marker))+").write_text(str(os.getpid()));time.sleep(60)"
            parent="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(worker)+"]);time.sleep(.12)"
            original=study.drain_owned
            def short(owner,known):
                return original(owner,known,((signal.SIGINT,.25),(signal.SIGTERM,.25),(signal.SIGKILL,2)))
            with patch.object(study,'drain_owned',side_effect=short):
                study.run_child([sys.executable,'-c',parent],log,{})
            self.assertTrue(marker.exists())
            self.assertIsNone(study.process_identity(int(marker.read_text())))
            self.assertEqual(json.loads(log.with_suffix('.cleanup.json').read_text())['remaining'],[])

    def test_failure_publishing_identity_still_drains_child(self):
        with tempfile.TemporaryDirectory() as d:
            original=study.publish;seen=[]
            def fail_once(path,value):
                if str(path).endswith('.process.json'):
                    seen.append(value['owner']);raise OSError('simulated metadata failure')
                return original(path,value)
            with patch.object(study,'publish',side_effect=fail_once),self.assertRaisesRegex(OSError,'metadata failure'):
                study.run_child([sys.executable,'-c','import time;time.sleep(60)'],Path(d)/'child.log',{})
            self.assertIsNone(study.process_identity(seen[0]['pid']))


if __name__=='__main__':unittest.main()
