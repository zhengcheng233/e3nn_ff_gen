#!/usr/bin/env python
import os
import h5py 
import numpy as np
from e3_layers.data import Batch
from glob import glob 
"""
generate hdf5 files for e3_layers from deepmd format
set the dir place as you like
"""

dirs = glob('/root/data.all/*/*/02.fp/data.*/set.000')

def dumpdata(dir0):
    #data = h5py.File(dir0,'r')
    #coord = data['coord']; energy = data['energy']
    #forces = data['force']; species = data['species']
    coord = np.load('set.000/coord.npy'); energy = np.load('set.000/energy.npy')
    forces = np.load('set.000/force.npy')
    type_map = ['H','C','N','O','S']
    type_n = np.loadtxt('type.raw')
    species = [type_map[int(u)] for u in type_n]
    sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
    species_n = np.array([sym_dict[s] for s in species]).reshape(-1,1)
    lst = []
    for i in range(len(coord)):
        c = coord[i]; e = energy[i]
        f = forces[i]#s = species[i].reshape(-1,1)
        num = len(c)
        #for i in range(len(species_n)):
        #    if species_n[i][0] > 0:
        #        num += 1
        lst += [ dict(pos=c[0:num],energy=e,forces=f[0:num],species=species_n)]
    path = 'fp.hdf5'
    attrs = {}
    attrs['pos'] = ('node', '1x1o')
    attrs['species'] = ('node', '1x0e')
    attrs['energy'] = ('graph', '1x0e')
    attrs['forces'] = ('node', '1x1o')

    batch = Batch.from_data_list(lst, attrs)
    batch.dumpHDF5(path)

pwd = os.getcwd()
for i in range(len(dirs)):
    os.chdir(dirs[i][:-7])
    f_name = os.path.basename(dirs[i])
    os.system('rm '+str(f_name))
    dir0 = os.path.basename(dirs[i])
    dumpdata(dir0)
    os.chdir(pwd)


