#!/usr/bin/env python

import numpy as np
import h5py

coulomb = np.load('coulomb1.npz',allow_pickle=True)
h5pyfile = h5py.File('protein_cc.hdf5','r')

E_coul = coulomb['E']; cc_coord = h5pyfile['coord']
cc_E = h5pyfile['energy']; cc_z = h5pyfile['species']
R_coul = coulomb['R']
disindex = np.load('disindex.npz')
index = disindex['arg']; dis = disindex['dis']

E = []; R = []; Species = []; E_delta = []
for i in range(len(index)):
    if dis[i] < 0.0001:
        if E_coul[i] == None:
            pass
        else:
            e_coul = E_coul[i]*27.2114; e_cc = cc_E[index[i]]
            r = cc_coord[index[i]]; spec = cc_z[index[i]]
            R.append(r); E.append(e_cc); Species.append(spec)
            E_delta.append(e_cc - e_coul)

f = h5py.File('protein_pure_cc.hdf5','w')
f.create_dataset('coord',data=R,dtype=np.single)
f.create_dataset('species',data=Species,dtype=np.intc)
f.create_dataset('energy',data=E,dtype=np.single)

f = h5py.File('protein_delta_cc.hdf5','w')
f.create_dataset('coord',data=R,dtype=np.single)
f.create_dataset('species',data=Species,dtype=np.intc)
f.create_dataset('energy',data=E_delta,dtype=np.single)
        
