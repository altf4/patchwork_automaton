"""Versioned numeric rollout files with indexed, bounded-memory random access.

Array payloads are contiguous, uncompressed bytes. One fixed-width footer index
stores offsets/shapes; scalar behavior fields retain their original precision.
Readers use pread and return owned arrays, so records survive reader closure and
use the existing decoded-cache accounting. No object pointers enter the file.
"""
import json
import os
from pathlib import Path
import struct

import numpy as np

MAGIC = b'PWROLL01'
HEADER = struct.Struct('<8sQ')
SCALAR_RECORD = struct.Struct('<qd??')
SCALARS = np.dtype([('action','<i8'),('value','<f8'),('advance','?'),('has_log_probs','?')])


class IndexedTrajectoryWriter:
    def __init__(self, path):
        path = Path(path)
        name = str(path)
        self.path = Path(name[:-7]+'.rlt' if name.endswith('.pkl.gz') else name+'.rlt')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file = self.path.open('w+b',buffering=1024*1024)
        self.file.write(HEADER.pack(MAGIC,0))
        self.count = 0
        self.fields = None
        self.rows, self.scalars = [], []
        self.closed = self.failed = False

    def write(self, sample):
        if self.closed or self.failed: raise ValueError('Rollout writer is unavailable')
        try:
            names = list(sample.arrays)
            values = list(sample.arrays.values())+[sample.memory,sample.log_probs]
            if self.fields is None:
                self.fields = [dict(name=k,dtype=None,rank=None) for k in names+['@memory','@log_probs']]
            if names != [f['name'] for f in self.fields[:-2]]:
                raise ValueError('Rollout array keys changed')
            row = []
            for field,value in zip(self.fields,values):
                if value is None:
                    if field['name']!='@log_probs':raise ValueError('Missing numeric rollout array')
                    row.append((self.file.tell(),0,0,0));continue
                value = np.ascontiguousarray(value)
                if not 1<=value.ndim<=3 or value.dtype.kind not in 'biuf':
                    raise ValueError('Rollout arrays require numeric/bool dtype and rank 1–3')
                if field['dtype'] is None:field.update(dtype=value.dtype.str,rank=value.ndim)
                if (value.dtype.str,value.ndim)!=(field['dtype'],field['rank']):
                    raise ValueError('Rollout array dtype/rank changed')
                padding = -self.file.tell()%8
                if padding:self.file.write(b'\0'*padding)
                row.append((self.file.tell(),*value.shape,*([0]*(3-value.ndim))))
                self.file.write(memoryview(value).cast('B'))
            self.rows.append(row)
            self.scalars.append((sample.action,sample.value,sample.advance,sample.log_probs is not None))
            self.count+=1
        except BaseException:
            self.failed=True
            raise

    def close(self):
        if self.closed:return
        try:
            if self.failed:return  # Header remains uncommitted; readers reject it.
            data_end=self.file.tell()
            index=np.asarray(self.rows,dtype='<u8').reshape(self.count,len(self.fields or []),4)
            index_offset=self.file.tell();self.file.write(index.tobytes())
            scalars_offset=self.file.tell();self.file.write(np.asarray(self.scalars,dtype=SCALARS).tobytes())
            manifest_offset=self.file.tell()
            manifest=dict(version=1,count=self.count,fields=self.fields or [],data_end=data_end,
                          index_offset=index_offset,scalars_offset=scalars_offset)
            self.file.write(json.dumps(manifest,separators=(',',':')).encode())
            self.file.seek(0);self.file.write(HEADER.pack(MAGIC,manifest_offset))
        finally:
            self.file.close();self.closed=True


class IndexedTrajectoryReader:
    def __init__(self,path):
        self.fd=os.open(path,os.O_RDONLY)
        self.read_calls=self.read_bytes=0
        try:
            size=os.fstat(self.fd).st_size
            header=self._read(HEADER.size,0)
            magic,footer=HEADER.unpack(header)
            if magic!=MAGIC or not HEADER.size<=footer<size or size-footer>65536:
                raise ValueError('Invalid or incomplete indexed rollout header')
            manifest=json.loads(self._read(size-footer,footer))
            if manifest.get('version')!=1:raise ValueError('Unsupported indexed rollout version')
            self.count=manifest['count'];self.fields=manifest['fields']
            end=manifest['data_end'];index=manifest['index_offset'];scalars=manifest['scalars_offset']
            if (type(self.count)is not int or self.count<0 or not isinstance(self.fields,list)
                    or len(self.fields)>128 or not HEADER.size<=end==index<=scalars<=footer
                    or scalars-index!=self.count*len(self.fields)*4*8
                    or footer-scalars!=self.count*SCALARS.itemsize):
                raise ValueError('Invalid indexed rollout bounds')
            if self.count and ([f['name'] for f in self.fields[-2:]]!=['@memory','@log_probs']
                               or len({f['name'] for f in self.fields})!=len(self.fields)):
                raise ValueError('Invalid indexed rollout fields')
            self.data_end=end
            # Read index rows on demand: metadata RAM does not grow with all
            # decisions in a large rollout. The caller bounds open readers.
            self.index_offset,self.scalars_offset=index,scalars
            self.specs=[]
            for field in self.fields:
                rank=field['rank'] if field['rank'] is not None else 1
                dtype=np.dtype(field['dtype'] or '<f4')
                if type(rank)is not int or not 1<=rank<=3 or dtype.kind not in 'biuf':
                    raise ValueError('Invalid indexed rollout dtype/rank')
                self.specs.append((rank,dtype))
        except BaseException:
            self.close()
            raise

    def _read(self,length,offset):
        payload=os.pread(self.fd,length,offset)
        self.read_calls+=1;self.read_bytes+=len(payload)
        if len(payload)!=length:raise ValueError('Incomplete indexed rollout file')
        return payload

    def __len__(self):return self.count

    def get(self,position):
        if self.fd is None:raise ValueError('Indexed rollout reader is closed')
        if not 0<=position<self.count:raise IndexError(position)
        from .training import Sample
        width=len(self.fields)*4*8
        raw_index=self._read(width+(8 if position+1<self.count else 0),self.index_offset+position*width)
        row=np.frombuffer(raw_index,dtype='<u8',count=len(self.fields)*4).reshape(len(self.fields),4)
        action,value,advance,has_log_probs=SCALAR_RECORD.unpack(self._read(SCALAR_RECORD.size,self.scalars_offset+position*SCALAR_RECORD.size))
        start=int(row[0,0]);end=struct.unpack_from('<Q',raw_index,width)[0] if position+1<self.count else self.data_end
        if not HEADER.size<=start<=end<=self.data_end:raise ValueError('Invalid indexed rollout record bounds')
        payload=self._read(end-start,start)
        values=[]
        for i,(rank,dtype) in enumerate(self.specs):
            if i==len(self.fields)-1 and not has_log_probs:
                values.append(None);continue
            offset=int(row[i,0]);shape=tuple(map(int,row[i,1:rank+1]))
            length=dtype.itemsize
            for dimension in shape:length*=dimension
            if not start<=offset<=offset+length<=end:raise ValueError('Invalid indexed rollout array bounds')
            # Own each output just like a decoded pickle; cache limits continue
            # to count all retained data and no mmap lifetime leaks into PPO.
            values.append(np.ndarray(shape,dtype=dtype,buffer=payload,offset=offset-start).copy())
        return Sample(dict(zip((f['name'] for f in self.fields[:-2]),values[:-2])),
                      values[-2],advance,action,value,values[-1])

    def __iter__(self):
        for position in range(self.count):yield self.get(position)

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
