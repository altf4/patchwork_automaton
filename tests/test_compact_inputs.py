"""The compact representation must reconstruct exact original tensor bytes."""
import unittest
import numpy as np
from patchwork_automaton.neural.compact_inputs import compact_array,SparseInput


class CompactInputTests(unittest.TestCase):
    def decode(self,packed):
        if not isinstance(packed,SparseInput):return packed
        out=np.zeros(packed.shape,dtype=packed.values.dtype)
        out.reshape(-1)[packed.indices[:,0]]=packed.values
        return out

    def test_exact_bits_including_signed_zero_nan_payload_and_infinity(self):
        bits=np.zeros(4096,np.uint32)
        bits[:7]=[0x80000000,0x7fc00001,0x7fc12345,0xff800000,0x7f800000,0x3f800000,1]
        source=bits.view(np.float32)
        packed=compact_array(source)
        self.assertIsInstance(packed,SparseInput)
        self.assertEqual(packed.indices.dtype,np.int32)
        self.assertEqual(source.tobytes(),self.decode(packed).tobytes())
        self.assertLess(packed.nbytes,source.nbytes/100)
        self.assertFalse(packed.values.flags.writeable)

    def test_dense_small_empty_scalar_and_bool_arrays_remain_exact(self):
        for source in [np.arange(4096,dtype=np.float32),np.zeros((2,0),np.float32),
                       np.asarray(-0.,np.float32),np.zeros(100,bool),np.zeros(12,np.int32)]:
            packed=compact_array(source)
            self.assertIsInstance(packed,np.ndarray)
            self.assertEqual(source.shape,packed.shape)
            self.assertEqual(source.dtype,packed.dtype)
            self.assertEqual(source.tobytes(),packed.tobytes())

    def test_noncontiguous_and_integer_arrays_roundtrip(self):
        for dtype in (np.float32,np.float64,np.int32,np.int64,np.uint32,np.uint64):
            source=np.zeros((128,64),dtype=dtype)[::2,::2]
            source[3,4]=7
            packed=compact_array(source)
            self.assertIsInstance(packed,SparseInput)
            self.assertEqual(source.tobytes(),self.decode(packed).tobytes())
            source[3,4]=8
            self.assertEqual(self.decode(packed)[3,4],7)

    def test_all_zero_payload_and_unique_ordered_indices(self):
        packed=compact_array(np.zeros((64,64),np.float32))
        self.assertIsInstance(packed,SparseInput)
        self.assertEqual(packed.nbytes,0)
        self.assertEqual(packed.indices.shape,(0,1))
        source=np.zeros(4096,np.float32);source[[4,100,300]]=1
        packed=compact_array(source)
        np.testing.assert_array_equal(packed.indices[:,0],[4,100,300])


if __name__=='__main__':unittest.main()
