"""Learner placement fails closed, while CPU services stay off learner GPUs."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class DeviceRoutingTests(unittest.TestCase):
    def test_gpu_requires_a_real_device_and_enables_growth_before_use(self):
        import tensorflow as tf
        from patchwork_automaton.neural.devices import configure_runtime
        with patch.object(tf.config, 'list_physical_devices', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'no GPU; CPU fallback'):
                configure_runtime('gpu')
        gpu = SimpleNamespace(name='/physical_device:GPU:0')
        with (patch.object(tf.config, 'list_physical_devices', return_value=[gpu]),
              patch.object(tf.config.experimental, 'set_memory_growth') as growth,
              patch.object(tf.config, 'set_soft_device_placement') as placement,
              patch.object(tf.config.experimental, 'enable_tensor_float_32_execution') as precision):
            self.assertEqual(configure_runtime('gpu'), '/GPU:0')
            growth.assert_called_once_with(gpu, True)
            placement.assert_called_once_with(False)
            precision.assert_called_once_with(False)

    def test_cpu_replica_masks_cuda_even_after_tensorflow_was_imported(self):
        import tensorflow as tf
        from patchwork_automaton.neural.devices import configure_runtime
        with (patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
              patch.object(tf.config, 'set_visible_devices') as visibility,
              patch.object(tf.config, 'set_soft_device_placement'),
              patch.object(tf.config.experimental, 'enable_tensor_float_32_execution')):
            self.assertEqual(configure_runtime('cpu', hide_gpu=True), '/CPU:0')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')
            visibility.assert_called_once_with([], 'GPU')
        with (patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
              patch.object(tf.config, 'set_visible_devices', side_effect=RuntimeError('already initialized'))):
            with self.assertRaisesRegex(RuntimeError, 'before TensorFlow device initialization'):
                configure_runtime('cpu', hide_gpu=True)

    def test_runtime_checks_reject_wrong_storage_and_returned_tensor_devices(self):
        from patchwork_automaton.neural.devices import assert_model_device, assert_output_device, canonical_device
        fake = SimpleNamespace(variables=[SimpleNamespace(path='weights', value=SimpleNamespace(device='/device:CPU:0'))])
        self.assertEqual(assert_model_device(fake, 'cpu'), {'weights': '/device:CPU:0'})
        with self.assertRaisesRegex(RuntimeError, 'Model variables must be on /GPU:0'):
            assert_model_device(fake, 'gpu')
        with self.assertRaisesRegex(RuntimeError, 'output tensors must be on /GPU:0'):
            assert_output_device({'logits': SimpleNamespace(device='/device:CPU:0')}, 'gpu')
        with self.assertRaisesRegex(RuntimeError, 'Model variables'):
            assert_model_device(SimpleNamespace(variables=[]), 'cpu')
        with self.assertRaisesRegex(ValueError, 'cpu or gpu'):
            canonical_device('automatic')

    def test_explicit_cpu_batcher_places_initial_trace_and_executor_results(self):
        import keras
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.batching import InferenceBatcher, example_batch
        from patchwork_automaton.neural.devices import device_scope

        class Toy(keras.Model):
            memory_size = 4

            def __init__(self):
                super().__init__()
                self.scale = self.add_weight(shape=(), initializer='ones')

            def call(self, inputs, training=False):
                return dict(logits=tf.cast(inputs['candidate_mask'], tf.float32)*self.scale,
                            value=inputs['memory_in'][:, :1]*self.scale,
                            memory_out=inputs['memory_in']+self.scale)

        async def run():
            with device_scope('cpu'):
                model = Toy()
            service = InferenceBatcher(model, max_batch=1, wait_ms=0, device='cpu')
            try:
                await service.start()
                output = await service._request(example_batch(4), 0, ('test', 'cpu'))
                np.testing.assert_array_equal(output['logits'], np.ones((1, 1), np.float32))
                np.testing.assert_array_equal(output['memory_out'], np.ones((1, 4), np.float32))
                stats = service.stats()
                self.assertEqual(stats['device'], '/CPU:0')
                self.assertTrue(all(value.endswith('/device:CPU:0') for value in stats['output_placements'].values()))
                self.assertEqual(stats['graph_traces'], 1)
                self.assertEqual(stats['requests'], 1)
            finally:
                await service.close()
        asyncio.run(run())
