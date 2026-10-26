import tempfile
from pathlib import Path
import unittest
from examples.analyze_ppo_update_profile import duration,intersect,invert,nvidia_samples,gpu_events,bin_occupancy
from test_neural import AVAILABLE


class TimingAnalysisTests(unittest.TestCase):
    def test_nested_and_concurrent_activity_uses_union_and_complement(self):
        active=[(1,3),(2,4),(6,7)]
        self.assertEqual(duration(active),4)
        self.assertEqual(invert(active,0,8),[[0,1],[4,6],[7,8]])
        self.assertEqual(intersect(active,[(0,2),(3,6.5)]),[[1,2],[3,4],[6,6.5]])
        self.assertEqual(duration(active)+duration(invert(active,0,8)),8)
        self.assertAlmostEqual(sum(bin_occupancy(active,0,8,8))/100,4)

    def test_gpu_csv_distinguishes_unavailable_from_zero_and_converts_clock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gpu.csv';p.write_text('2026/10/25 16:00:00.000, 0, N/A, 2000, 24000, 50, 900, 5001\n')
            rows=nvidia_samples(p,1792969200_000000000)
            self.assertEqual(rows[0]['elapsed_seconds'],0)
            self.assertEqual(rows[0]['gpu_busy_percent'],0)
            self.assertIsNone(rows[0]['memory_busy_percent'])

    @unittest.skipUnless(AVAILABLE,'Requires TensorFlow protobufs')
    def test_gpu_derived_rows_do_not_fill_idle_gaps(self):
        from tensorflow.tsl.profiler.protobuf import xplane_pb2
        space=xplane_pb2.XSpace();plane=space.planes.add();plane.name='/device:GPU:0'
        plane.event_metadata[1].name='gemm';plane.event_metadata[2].name='MemcpyHtoD';plane.event_metadata[3].name='Derived scope'
        for identity,name,event_id,start,length in [(7,'Stream #7',1,0,1),(8,'Stream #8 (MemcpyHtoD)',2,2,1),(-1,'TensorFlow Ops',3,0,10)]:
            line=plane.lines.add();line.id=identity;line.name=name;line.timestamp_ns=1_000_000_000
            event=line.events.add();event.metadata_id=event_id;event.offset_ps=start*10**12;event.duration_ps=length*10**12
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.xplane.pb';p.write_bytes(space.SerializeToString());events,planes,errors=gpu_events([p])
            self.assertEqual(len(events),2)
            self.assertEqual(duration((e['start'],e['end']) for e in events),2)
            self.assertEqual({e['kind'] for e in events},{'kernel','copy'})
            self.assertFalse(planes[0]['lines'][-1]['raw_stream_selected'])
            environment=space.planes.add();environment.name='Task Environment';environment.stat_metadata[1].name='profile_start_time'
            stat=environment.stats.add();stat.metadata_id=1;stat.uint64_value=1_000_000_000_000
            p.write_bytes(space.SerializeToString());aligned,_,_=gpu_events([p])
            self.assertEqual(aligned[0]['start'],1001.0)


if __name__=='__main__':unittest.main()
