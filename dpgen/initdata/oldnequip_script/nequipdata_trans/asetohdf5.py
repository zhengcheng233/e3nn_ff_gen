#!/usr/bin/env python
import h5py
import os 
import torch
import numpy as np
from tqdm import tqdm
from nequip.data import AtomicData,AtomicDataDict

from ase import Atoms
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

table = ['H','He','Li','Be','B','C','N','O','F','Ne']
table += ['Na','Mg','Al','Si','P','S','Cl','Ar']

symbol2idx = {table[i]:i+1 for i in range(len(table))}
path_ase = './protein_mp2_delta.xyz'
path_hdf5 = 'protein_mp2_delta.hdf5'
use_forces = False

mols = read(path_ase,index=':')
cnt = len(mols)
max_size = max([len(item) for item in mols])

with h5py.File(path_hdf5,'w') as f:
    f.create_dataset('coord',data=np.zeros((cnt,max_size,3),dtype=np.single))
    f.create_dataset('species',data=np.zeros((cnt,max_size),dtype=np.intc))
    f.create_dataset('energy',data=np.zeros((cnt,),dtype=np.single))
    if use_forces:
        f.create_dataset('forces',data=np.zeros((cnt,max_size,3),dtype=np.single))
    for i,mol in enumerate(tqdm(mols)):
        cur_size = len(mol)
        symbols = np.zeros((max_size,),dtype=int)
        f['coord'][i,:cur_size] = mol.get_positions()
        f['species'][i,:cur_size] = mol.get_atomic_numbers()
        f['energy'][i] = mol.get_total_energy()
        if use_forces:
            f['forces'][i,:cur_size] = mol.get_forces()
