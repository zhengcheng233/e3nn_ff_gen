#!/usr/bin/env python
import os
import h5py 
import numpy as np
from e3_layers.data import Batch
from glob import glob 
import ase
"""
generate hdf5 files for e3_layers from gaussian log file
"""

#dirs = glob('/root/data.all/*/*/02.fp/data.*/set.000')

dirs = glob('./init/data*/iter.*/02.fp/data.*')
atomic_e_adz = {'H':-0.5022689,'C':-37.8373345925,'N':-54.5765396405,'O':-75.0510133225,'S':-398.103592966}
symbols = ase.data.chemical_symbols

def dumpdata(coord,symbol,E,F):
    #coord = np.load('set.000/coord.npy'); energy = np.load('set.000/energy.npy')
    #forces = np.load('set.000/force.npy')
    #type_map = ['H','C','N','O','S']
    #type_n = np.loadtxt('type.raw')
    #species = [type_map[int(u)] for u in type_n]
    os.system('rm fp.hdf5')
    species = symbol[0]
    sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
    species_n = np.array([sym_dict[s] for s in species]).reshape(-1,1)
    
    lst = []
    for i in range(len(coord)):
        c = coord[i]; e = E[i]
        f = F[i]#s = species[i].reshape(-1,1)
        num = len(c)
        #for i in range(len(species_n)):
        #    if species_n[i][0] > 0:
        #        num += 1
        lst += [ dict(pos=c[0:num],energy=[e],forces=f[0:num],species=species_n)]
    path = 'fp.hdf5'
    attrs = {}
    attrs['pos'] = ('node', '1x1o')
    attrs['species'] = ('node', '1x0e')
    attrs['energy'] = ('graph', '1x0e')
    attrs['forces'] = ('node', '1x1o')

    batch = Batch.from_data_list(lst, attrs)
    batch.dumpHDF5(path)

def exactlog(f0):
    file_name = os.path.join(f0,'input.log')
    energy_t = []; coords_t = []; atom_symbols = []; forces_t = []
    atomic_e_adz = {'H':-0.5022689,'C':-37.8373345925,'N':-54.5765396405,'O':-75.0510133225,'S':-398.103592966}
    flag = 0
    with open(file_name) as fp:
        for line in fp:
            if line.startswith(" SCF Done"):
                # energies
                energy = float(line.split()[4])
            elif line.startswith(" Center     Atomic                   Forces (Hartrees/Bohr)"):
                flag = 1
                forces = []
            elif line.startswith("                          Input orientation:") or line.startswith("                         Z-Matrix orientation:"):
                flag = 5
                coords = []
                atom_symbols = []

            if 1 <= flag <= 3 or 5 <= flag <= 9:
                flag += 1
            elif flag == 4:
                # forces
                if line.startswith(" -------"):
                    forces_t.append(forces)
                    energy_t.append(energy)
                    coords_t.append(coords)
                    flag = 0
                else:
                    s = line.split()
                    forces.append([float(line[23:38]), float(line[38:53]), float(line[53:68])])
            elif flag == 10:
                # atom_symbols and coords
                if line.startswith(" -------"):
                    flag = 0
                else:
                    s = line.split()
                    coords.append([float(x) for x in s[3:6]])
                    atom_symbols.append(symbols[int(s[1])])
    if len(energy_t) == 0: 
        print('*************')
        print('unnormal')
        print(os.getcwd())
        print(f0)
    for ele in atom_symbols:
        energy_t[0] -= atomic_e_adz[ele]
    return coords_t[0], atom_symbols, energy_t[0] * 27.2113966, np.array(forces_t[0]) * 27.2113966 / 0.52918


def fail_check():
    """
    add fail case check to avoid gaussian task not converged
    """
    log_file = glob('./property_*')
    coord = []; symbol = []; E = []; F = []
    for f0 in log_file:
        c,s,e,f = exactlog(f0)
        if e == None or len(f) == 0:
            print('check')
        else:
            coord.append(c); symbol.append(s)
            E.append(e); F.append(f)
    return coord,symbol,E,F

pwd = os.getcwd()
#for i in range(10):
for i in range(len(dirs)):
    os.chdir(dirs[i])
    #print(os.getcwd())
    coord,symbol,E,F = fail_check()
    #print(E)
    #f_name = os.path.basename(dirs[i])
    #os.system('rm '+str(f_name))
    #dir0 = os.path.basename(dirs[i])
    if len(E) > 0:
        dumpdata(coord,symbol,E,F)
    os.chdir(pwd)


