#!/usr/bin/env python
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

data = np.load('solvated_protein_fragments.npz',allow_pickle=True)
E = data['E']; R = data['R']; Z = data['Z']; F = data['F']; N = data['N']
sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}

for i in range(len(E)):
    if E[i] == None:
        pass
    else:
        n_atom = N[i]
        symbol = []; z = Z[i][0:n_atom]
        for j in range(len(z)):
            symbol.append(sym_dict[z[j]])
        #force = np.array(R[i])*0.
        atoms = Atoms(positions=R[i][0:n_atom],symbols=symbol,pbc=False)
        calculator = SinglePointCalculator(atoms,energy=E[i],forces=F[i][0:n_atom])
        atoms.calc = calculator
        write('protein_dft.xyz',atoms,format='extxyz',append=True)
