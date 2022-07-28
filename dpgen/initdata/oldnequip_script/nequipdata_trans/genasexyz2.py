#!/usr/bin/env python
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

data = np.load('mp2cbs.npz',allow_pickle=True)
E = data['Emp2cbs']; R = data['R']; Z = data['Z']
sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}
atomic_mp2 = {1:-0.499395*27.2114,6:-37.69279286*27.2114,7:-54.4031059*27.2114,8:-74.81696847*27.2114,16:-397.720405*27.2114}
for i in range(len(E)):
    if E[i] == None:
        pass
    else:
        e = E[i]*27.2114
        symbol = []; z = Z[i]
        for j in range(len(z)):
            symbol.append(sym_dict[z[j]])
            e -= atomic_mp2[z[j]]
        force = np.array(R[i])*0.
        atoms = Atoms(positions=R[i],symbols=symbol,pbc=False)
        calculator = SinglePointCalculator(atoms,energy=e,forces=force)
        atoms.calc = calculator
        write('protein_mp2.xyz',atoms,format='extxyz',append=True)