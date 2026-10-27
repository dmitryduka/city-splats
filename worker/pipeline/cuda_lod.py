"""GPU voxel hierarchy with all original leaf splats, encoded as native Spark RAD.

The hierarchy differs from Spark's serial greedy Bhattacharyya builder. This is
recorded explicitly in metrics; leaf geometry, colors and SH are retained.
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import cupy as cp
import numpy as np
from plyfile import PlyData

_GROUP_MEAN = cp.RawKernel(r'''
extern "C" __global__ void group_mean(
    const float* values, const float* weights, const long long* starts,
    const long long* counts, int groups, int dims, float* result, float* mass) {
    long long item = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (item >= (long long)groups * dims) return;
    int g = item / dims, d = item % dims;
    double sum = 0.0, total = 0.0;
    for (long long j = starts[g]; j < starts[g] + counts[g]; ++j) {
        double w = weights[j];
        total += w;
        sum += w * values[j * dims + d];
    }
    result[item] = (float)(sum / total);
    if (d == 0) mass[g] = (float)total;
}
''', 'group_mean')


def _load(ply):
    vertices = PlyData.read(ply,mmap='r')['vertex'].data
    field = lambda names: cp.asarray(np.column_stack([vertices[n] for n in names]),dtype=cp.float32)
    xyz = field(['x','y','z'])
    if 'red' in vertices.dtype.names:
        raise ValueError('LoD expects Gaussian PLY; convert the dense cloud to splats first')
    scales = cp.exp(field([f'scale_{i}' for i in range(3)]))
    alpha = 1/(1+cp.exp(-cp.asarray(vertices['opacity'].copy())))
    quat = field([f'rot_{i}' for i in (1,2,3,0)])
    quat /= cp.linalg.norm(quat,axis=1)[:,None]
    quat *= cp.where(quat[:,3:4]<0,-1,1)
    rgb = field([f'f_dc_{i}' for i in range(3)])*.28209479177387814+.5
    rest = sorted((n for n in vertices.dtype.names if n.startswith('f_rest_')),key=lambda n:int(n[7:]))
    sh = field(rest).reshape(len(xyz),3,-1).transpose(0,2,1).reshape(len(xyz),-1) if rest else None
    return dict(xyz=xyz,rgb=rgb,scales=scales,alpha=alpha,quat=quat,sh=sh,
                count=cp.zeros(len(xyz),dtype=cp.uint16),start=cp.zeros(len(xyz),dtype=cp.uint32))


def _hierarchy(current):
    levels = []
    step = max(float(cp.median(current['scales'].max(1)))*4,1e-6)
    origin = current['xyz'].min(0)
    for iteration in range(80):
        n = len(current['xyz'])
        if n==1:
            levels.append(current)
            return levels
        cells = cp.floor((current['xyz']-origin)/step).astype(cp.int64)
        size = cells.max(0)+1
        if float(cp.prod(size.astype(cp.float64)))>=2**62:
            step *= 2
            continue
        keys = (cells[:,0]*size[1]+cells[:,1])*size[2]+cells[:,2]
        order = cp.argsort(keys)
        keys = keys[order]
        starts = cp.r_[cp.int64(0),cp.flatnonzero(keys[1:]!=keys[:-1])+1]
        counts = cp.diff(cp.r_[starts,cp.int64(n)])
        maximum = int(counts.max())
        if maximum>65535:
            rank = cp.arange(n)-cp.repeat(starts,counts)
            keys = keys*math.ceil(maximum/65535)+rank//65535
            starts = cp.r_[cp.int64(0),cp.flatnonzero(keys[1:]!=keys[:-1])+1]
            counts = cp.diff(cp.r_[starts,cp.int64(n)])
        if len(starts)==n:
            step *= 2
            continue
        current = {k:(v[order] if v is not None else None) for k,v in current.items()}
        levels.append(current)
        # Area/opacity weighting retains each group's photometric mass.
        scales = current['scales']
        area = ((scales[:,0]*scales[:,1])**1.6075+(scales[:,0]*scales[:,2])**1.6075+(scales[:,1]*scales[:,2])**1.6075)
        area = (area/3)**(1/1.6075)
        weight = cp.maximum(area*current['alpha'],1e-20)
        # Per-group double accumulation avoids cumsum cancellation across tiny and huge splats.
        total = cp.empty(len(starts),dtype=cp.float32)
        def mean(a):
            a=cp.ascontiguousarray(a,dtype=cp.float32)
            result=cp.empty((len(starts),a.shape[1]),dtype=cp.float32)
            _GROUP_MEAN(((result.size+255)//256,),(256,),
                        (a,weight,starts,counts,np.int32(len(starts)),np.int32(a.shape[1]),result,total))
            return result
        xyz,rgb = mean(current['xyz']),mean(current['rgb'])
        parent_of = cp.repeat(cp.arange(len(starts)),counts)
        # Isotropic parents enclose all children; original anisotropic leaves stay intact.
        extent = cp.linalg.norm(current['xyz']-xyz[parent_of],axis=1)+3*scales.max(1)
        radius = cp.zeros(len(starts),dtype=extent.dtype)
        cp.maximum.at(radius,parent_of,extent)
        radius /= 3
        parent_scales = cp.repeat(radius[:,None],3,axis=1)
        parent_alpha = cp.clip(total/cp.maximum(radius*radius,1e-20),.01,1.)
        current = dict(xyz=xyz,rgb=rgb,scales=parent_scales,alpha=parent_alpha,
                       quat=cp.tile(cp.array([0,0,0,1],dtype=cp.float32),(len(starts),1)),
                       sh=mean(current['sh']) if current['sh'] is not None else None,
                       count=counts.astype(cp.uint16),start=starts.astype(cp.uint32))
        print(f'CUDA LoD level {iteration}: {n} -> {len(xyz)}',flush=True)
        step *= 2
    raise RuntimeError('CUDA LoD hierarchy did not converge')


def _pad(data):
    return data+b'\0'*((-len(data))%8)


def _write_chunk(path,base,node,child_offset,max_sh):
    props,payload = [],bytearray()
    def add(name,array,encoding,dtype):
        array = cp.asarray(array).astype(dtype)
        if array.ndim==2:
            array = array.T
        raw = cp.asnumpy(cp.ascontiguousarray(array)).tobytes()
        props.append(dict(property=name,encoding=encoding,offset=len(payload),bytes=len(raw)))
        payload.extend(_pad(raw))
    add('center',node['xyz'],'f32',cp.float32)
    add('alpha',node['alpha'],'f32',cp.float32)
    add('rgb',node['rgb'],'f16',cp.float16)
    add('scales',node['scales'],'f32',cp.float32)
    add('orientation',node['quat'][:,:3],'f32',cp.float32)
    if max_sh:
        for degree,(a,b) in enumerate([(0,9),(9,24),(24,45)],1):
            if degree<=max_sh:
                add(f'sh{degree}',node['sh'][:,a:b],'f16',cp.float16)
    add('child_count',node['count'],'u16',cp.uint16)
    add('child_start',cp.where(node['count']>0,node['start']+child_offset,0),'u32',cp.uint32)
    meta=json.dumps(dict(version=1,base=base,count=len(node['xyz']),payloadBytes=len(payload),
                         maxSh=max_sh,lodTree=True,properties=props),separators=(',',':')).encode()
    data=struct.pack('<II',0x43444152,len(meta))+_pad(meta)+struct.pack('<Q',len(payload))+payload
    path.write_bytes(data)
    return len(data)


def build_lod(ply,out_dir):
    started=time.perf_counter()
    nodes=_load(ply)
    leaves=len(nodes['xyz'])
    sh_size=nodes['sh'].shape[1] if nodes['sh'] is not None else 0
    max_sh={0:0,9:1,24:2,45:3}[sh_size]
    levels=_hierarchy(nodes)
    del nodes
    levels.reverse()
    counts=[len(level['xyz']) for level in levels]
    offsets=np.cumsum([0]+counts).tolist()
    if offsets[-1]>=2**32:
        raise ValueError('RAD child indices exceed uint32')
    # Chunk boundaries are global multiples of 65536, including across hierarchy levels.
    chunks=[]
    pending=[]
    pending_count=0
    base=0
    for level_index,level in enumerate(levels):
        child_offset=offsets[level_index+1] if level_index+1<len(levels) else 0
        level['start']=cp.where(level['count']>0,level['start']+child_offset,0).astype(cp.uint32)
        first=0
        while first<len(level['xyz']):
            take=min(65536-pending_count,len(level['xyz'])-first)
            pending.append({k:(v[first:first+take] if v is not None else None) for k,v in level.items()})
            first+=take;pending_count+=take
            if pending_count==65536:
                node={k:(cp.concatenate([p[k] for p in pending]) if level[k] is not None else None) for k in level}
                filename=f'splat-lod-{len(chunks)}.radc'
                size=_write_chunk(out_dir/filename,base,node,0,max_sh)
                chunks.append(dict(offset=0,bytes=size,filename=filename))
                base+=pending_count;pending=[];pending_count=0
    if pending:
        node={k:(cp.concatenate([p[k] for p in pending]) if pending[0][k] is not None else None) for k in pending[0]}
        filename=f'splat-lod-{len(chunks)}.radc'
        size=_write_chunk(out_dir/filename,base,node,0,max_sh)
        chunks.append(dict(offset=0,bytes=size,filename=filename))
    meta=json.dumps(dict(version=1,type='gsplat',count=offsets[-1],maxSh=max_sh,lodTree=True,
                         chunkSize=65536,allChunkBytes=sum(c['bytes'] for c in chunks),chunks=chunks,
                         comment='CUDA voxel hierarchy; original leaf splats retained'),separators=(',',':')).encode()
    rad=out_dir/'splat-lod.rad'
    rad.write_bytes(struct.pack('<II',0x30444152,len(meta))+_pad(meta))
    cp.cuda.get_current_stream().synchronize()
    metrics=dict(backend='cuda_voxel_hierarchy',leaf_splats=leaves,total_nodes=offsets[-1],
                 seconds=time.perf_counter()-started,baseline_algorithm='spark_bhattacharyya',
                 same_lod_algorithm=False,leaf_geometry_preserved=True)
    (out_dir/'cuda-lod.json').write_text(json.dumps(metrics,indent=2))
    return rad
