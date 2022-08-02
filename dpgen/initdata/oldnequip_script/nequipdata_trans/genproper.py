#!/usr/bin/env python

import h5py
from tqdm import tqdm
import numpy as np

from functools import lru_cache
from e3nn import o3
import torch
import pdb
import pickle
import os

data = np.load('solvated_protein_fragments.npz',allow_pickle=True)
R = data['R']; Z = data['Z']; D = data['D']
cnt = len(Z)
max_size = len(Z[0])

with h5py.File('moldipole.hdf5','w') as f:
    f.create_dataset("coord",data=np.zeros((cnt,max_size,3),dtype=np.single))
    f.create_dataset("species",data=np.zeros((cnt,max_size),dtype=np.intc))
    f.create_dataset("dipole",data=np.zeros((cnt,1,3),dtype=np.single))
    for i in range(cnt):
        f['coord'][i] = R[i]
        f['species'][i] = Z[i]
        f['dipole'][i] = D[i]
