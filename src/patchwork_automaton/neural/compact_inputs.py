"""Lossless storage for zero-heavy inputs; no model or feature transformation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SparseInput:
    shape: tuple
    indices: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self):
        return self.indices.nbytes+self.values.nbytes


def compact_array(array):
    """Keep every nonzero bit pattern, including negative zero and NaN payloads.

    Dense/small arrays stay dense. Indices are unique flat int32 positions, so
    reconstruction is assignment, never sparse reduction or floating addition.
    """
    source=np.asarray(array)
    array=np.ascontiguousarray(source).reshape(source.shape)
    if (array.nbytes<4096 or array.dtype.kind not in 'fiu'
            or array.dtype.itemsize not in (4,8) or not array.dtype.isnative
            or array.size>np.iinfo(np.int32).max):
        return array
    bits=array.view(np.dtype(f'u{array.dtype.itemsize}')).reshape(-1)
    indices=np.flatnonzero(bits).astype(np.int32)
    if len(indices)*(4+array.dtype.itemsize)>=array.nbytes*.8:
        return array
    values=array.reshape(-1)[indices].copy()
    indices=indices[:,None]
    indices.flags.writeable=values.flags.writeable=False
    return SparseInput(array.shape,indices,values)


class DeviceSparseInput:
    """Only indices and exact values persist on the device between uses."""
    def __init__(self,packed,tf):
        self.shape=packed.shape
        self.indices=tf.convert_to_tensor(packed.indices)
        self.values=tf.convert_to_tensor(packed.values)

    def materialize(self,tf):
        # scatter_nd adds into zeros, which can discard a negative-zero sign.
        # Unique-index update writes the original bit patterns without sums.
        flat=tf.zeros([int(np.prod(self.shape))],dtype=self.values.dtype)
        flat=tf.tensor_scatter_nd_update(flat,self.indices,self.values)
        return tf.reshape(flat,self.shape)
