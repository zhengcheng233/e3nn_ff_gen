#!/usr/bin/env python
import os 
import numpy as np
import sys
import h5py
import torch
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
import ase
import os

atomic_e = {'H':-0.500607632539,'C':-37.8302344100,'N':-54.5680042961,'O':-75.0362322778,'S':-398.081113777}
coord = []; symbol = []; energy = None; force = []; atomic_mol = 0.
with open('input.com','r') as fp:
    for line in fp:
        line = line.strip().split()
        if len(line) == 4:
            symbol.append(line[0])
            coord.append([float(line[1]),float(line[2]),float(line[3])])
            atomic_mol += atomic_e[line[0]]

with open('input.log','r') as fp:
    flag = 0
    for line in fp:
        if line.startswith(" SCF Done"):
            energy = float(line.split()[4]) - atomic_mol
        elif line.startswith(" Center     Atomic                   Forces (Hartrees/Bohr)"):
            flag = 1
        if 1 <= flag <= 3:
            flag += 1
        elif flag == 4:
            if line.startswith(" -------"):
                flag = 0
            else:
                s = line.split()
                force.append([float(s[2]),float(s[3]),float(s[4])])

energy = energy * 27.21139664 
force = np.array(force) * 27.21139664 / 0.529177249

atomic_n = ase.atom.atomic_numbers
coord = np.array(coord,dtype=np.single)
species_n = [atomic_n[u] for u in symbol]
species_n = np.array(species_n, dtype=np.intc)
energy = np.array(energy,dtype=np.single)
force = np.array(force,dtype=np.single)

# static the model_devi; rmse_single and so on. 
np.savez('single.npz',coord=coord,species_n=species_n,energy=energy,force=force)
