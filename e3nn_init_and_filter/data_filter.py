#!/usr/bin/env python
"""
this scripts is used to filter unreasonable strustures using the reason.txt generate from topo_offline_filter.py
"""
import numpy as np
import os
from glob import glob
from e3_layers.data import Batch

type_map = [6,1,7,8,16]
dirs = glob('./iter.*/02.fp/data.*')
attrs =  {'pos':('node','1x1o'),'species': ('node','1x0e'), 'energy': ('graph', '1x0e'), 'forces': ('node', '1x1o')}

cwd_ = os.getcwd()
for dir0 in dirs:
    dir_n = dir0[2:]
    dir_n1 = os.path.join('./databack',dir_n)
    os.system('mkdir -p '+dir_n1)
    os.system('cp -r ' + dir_n + '/* ' + dir_n1)

dirs = glob('./databack/iter.*/02.fp/data.*')
for dir0 in dirs:
    os.chdir(dir0)
    os.system('rm *.hdf5')
    reasons = np.loadtxt('reason.txt')
    if len(reasons.shape) > 1:
        reasons = reasons[:,-1]
    else:
        reasons = [reasons[-1]]
    reasons = np.array([int(u) for u in reasons])
    reason_idx = np.where(reasons>0)[0]
    if len(reason_idx) < len(reasons):
        print('unreason')
        print(dir0)
    coord = np.load('set.000/coord.npy'); force = np.load('set.000/force.npy')
    energy = np.load('set.000/energy.npy')
    type_e = np.loadtxt('type.raw',dtype=int)
    symbol = [type_map[u] for u in type_e]
    coord = np.array(coord, dtype = np.single); force = np.array(force, dtype = np.single)
    energy = np.array(energy, dtype = np.single); symbol = np.array(symbol, dtype = np.intc)
    energy = energy[reason_idx]; coord = coord[reason_idx]
    force = force[reason_idx]
    if len(coord) > 0:
        lst = []
        [lst.append(dict(pos=coord[ii].reshape((len(symbol),3)),energy=energy[ii],forces=force[ii],species=np.array(symbol))) for ii in range(len(coord))]
        path = 'fp.hdf5'
        batch = Batch.from_data_list(lst, attrs); batch.dumpHDF5(path)
    os.chdir(cwd_)
