#!/usr/bin/env python

import h5py
import os
import torch
import numpy as np
from tqdm import tqdm
from nequip.data import AtomicData, AtomicDataDict


table = ['H','He','Li','Be','B','C','N','O','F','Ne']
table += ['Na','Mg','Al','Si','P','S','Cl','Ar']

symbol2idx = {table[i]:i+1 for i in range(len(table))}
data = np.load('solvated_protein_fragments.npz')
E = data['E']; F = data['F']; R = data['R']; Z = data['Z']

max_len = 0; cnt = len(Z[0])


f = h5py.File('proteindft.hdf5','w')
f.create_dataset('coord',data=R,dtype=np.single)
f.create_dataset('species',data=Z,dtype=np.intc)
f.create_dataset('energy',data=E,dtype=np.single)
f.create_dataset('force',data=F,dtype=np.single)


#print(E[0])
#print(F[0])
#print(R[0])
#print(Z[0])
