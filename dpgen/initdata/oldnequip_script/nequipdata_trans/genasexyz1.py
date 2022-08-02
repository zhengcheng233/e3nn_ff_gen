#!/usr/bin/env python
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

data = np.load('cccbs.npz',allow_pickle=True)
E = data['E']; R = data['R']; Z = data['Z']
sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}
atomic_ccsd = {1: -13.6024856805864, 6: -1028.173885158564, 7: -1483.75126621355, 8: -2040.8356246668582, 16: -10826.811568272991}

for i in range(len(E)):
    if E[i] == None:
        pass
    else:
        symbol = []; z = Z[i]; e = E[i]*27.2114
        for j in range(len(z)):
            symbol.append(sym_dict[z[j]])
            e -= atomic_ccsd[z[j]]
        force = np.array(R[i])*0.
        atoms = Atoms(positions=R[i],symbols=symbol,pbc=False)
        calculator = SinglePointCalculator(atoms,energy=e,forces=force)
        atoms.calc = calculator
        write('protein_ccsd.xyz',atoms,format='extxyz',append=True)
