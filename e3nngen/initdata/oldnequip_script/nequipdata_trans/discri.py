#!/usr/bin/env python

import numpy as np
import h5py 

coulomb = np.load('coulomb.npz',allow_pickle=True)
h5pyfile = h5py.File('protein_cc.hdf5','r')

R_coul = coulomb['R']
R_h5py = h5pyfile['coord']
Z_h5py = h5pyfile['species']
R0 = []; R1 = []
for i in range(len(R_coul)):
    r0 = R_coul[i] 
    r1 = R_h5py[i]#*0.529177249
    #r2 = np.zeros((len(r1),3))
    #r2[0:len(r0)] += r0
    z = Z_h5py[i]
    r2 = []
    for j in range(len(z)):
        if z[j] > 0.:
            r2.append(r1[j])
    R0.append(np.array(r0))
    R1.append(np.array(r2))

disarg = []; dismin = []
#for i in range(10):
for i in range(len(R0)):
    r0 = R0[i]
    dis = []
    for j in range(len(R1)):
        if len(r0) == len(R1[j]):
            dis0 = np.sum((r0-R1[j])**2)
            dis.append(dis0)
        else:
            dis.append(1000000.)
    dis = np.array(dis)
    disarg.append(np.argmin(dis))
    dismin.append(dis[np.argmin(dis)])
#print(dismin)
np.savez('disindex.npz',arg=disarg,dis=dismin)
 
