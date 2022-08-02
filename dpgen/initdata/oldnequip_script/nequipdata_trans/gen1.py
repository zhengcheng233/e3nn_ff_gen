#!/usr/bin/env python
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

data = np.load('cccbs.npz',allow_pickle=True)
E = data['E']; R = data['R']; Z = data['Z']
sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}
atomic_mp2 = {1:-0.499395*27.2114,6:-37.69279286*27.2114,7:-54.4031059*27.2114,8:-74.81696847*27.2114,16:-397.720405*27.2114}
atomic_dft = {1:-13.71793959,6:-1029.83166273,7:-1485.408061261,8:-2042.792034436,16:-10831.264715514}
atomic_ccsd = {1: -13.6173048, 6: -1028.173885158564, 7: -1483.75126621355, 8: -2040.8356246668582, 16: -10826.811568272991}
mono_ind = np.load('mono_ind.npy',allow_pickle=True); cc_ind = np.load('cc_index.npy',allow_pickle=True)
cc1_ind = np.load('ord_num.npy',allow_pickle=True)

E_cc = np.load('DFT_mono.npy',allow_pickle=True)
#E_cc = np.load('Emp2cbs_ord.npy',allow_pickle=True)
#print(np.max(E_cc)-np.min(E_cc))
E_delta = []
#for i in range(100):
for i in range(len(E)):
    if E[i] == None or E_cc[i] == None:
        pass
    else:
        symbol = []; z = Z[i]; e = E[i]*27.2114
        e_mp2 = E_cc[i]#*27.2114
        for j in range(len(z)):
            symbol.append(sym_dict[z[j]])
            #e_dft += atomic_dft[z[j]]
            e -= atomic_ccsd[z[j]]
            #e_mp2 -= atomic_mp2[z[j]]
        #print(e)
        #print(e_dft)
        #print(e-e_dft)
        E_delta.append(e)#e_mp2)
        #force = np.array(R[i])*0.
        #atoms = Atoms(positions=R[i],symbols=symbol,pbc=False)
        #calculator = SinglePointCalculator(atoms,energy=e-e_mp2,forces=force)
        #atoms.calc = calculator
        #write('protein_ccsd_delta1.xyz',atoms,format='extxyz',append=True)
E_delta = np.array(E_delta)
print(E_delta)
print(np.max(E_delta)-np.min(E_delta))
