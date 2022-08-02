#!/usr/bin/env python
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

data = np.load('mp2cbs.npz',allow_pickle=True)
E = data['Emp2cbs']; R = data['R']; Z = data['Z']
sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}
atomic_dft = {1:-13.71793959,6:-1029.83166273,7:-1485.408061261,8:-2042.792034436,16:-10831.264715514}
atomic_mp2 = {1:-0.499395*27.2114,6:-37.69279286*27.2114,7:-54.4031059*27.2114,8:-74.81696847*27.2114,16:-397.720405*27.2114}
mono_ind = np.load('mono_ind.npy',allow_pickle=True)

DFT = np.load('solvated_protein_fragments.npz',allow_pickle=True)
E_dft = DFT['E']; E_mono = E_dft[mono_ind]
#for i in range(1000):
for i in range(len(E)):
    if E[i] == None:
        pass
    else:
        e = E[i]*27.2114; e_dft = E_mono[i]
        symbol = []; z = Z[i]
        for j in range(len(z)):
            symbol.append(sym_dict[z[j]])
            e -= atomic_mp2[z[j]]
        force = np.array(R[i])*0.
        #print(e - e_dft)
        atoms = Atoms(positions=R[i],symbols=symbol,pbc=False)
        calculator = SinglePointCalculator(atoms,energy=e-e_dft,forces=force)
        atoms.calc = calculator
        write('protein_mp2_delta.xyz',atoms,format='extxyz',append=True)
