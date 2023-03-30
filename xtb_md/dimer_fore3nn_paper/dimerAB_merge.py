#!/usr/bin/env python
"""
this script is used to merge the data of dimer AB, all data in confA_B/*/gjfs/*/*/input.gjf and confA_B/*/gjfs/*/*/coord.npy should merge in one file 
"""
import ase
import os
from glob import glob 
import numpy as np

atomic_number = ase.data.atomic_numbers

confs = glob('./conf.*')
def merge_data():
    coords = []
    f_names = glob('./gjfs/*/*/input.gjf')
    for f_name in f_names:
        f_dir = os.path.dirname(f_name)
        f_name0 = os.path.join(f_dir,'input.gjf')
        f_name1 = os.path.join(f_dir,'coord.npy')
        coord = []
        with open(f_name0,'r') as fp:
            for line in fp:
                line = line.strip().split()
                if len(line) == 4:
                    coord.append([float(line[1]),float(line[2]),float(line[3])])
        coords.append(coord)
        coord1 = np.load(f_name1)
        coords.extend(coord1)
    return coords


cwd = os.getcwd()
for dir0 in confs:
    os.chdir(dir0)
    coords = merge_data()
    # here we can save coord as npy or as e3nn data file, may be with hdf5 file
    species_n = []; q_net = None
    with open('dimer_sapt.gjf','r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 2 and line[-1] == '1':
                q_net = int(line[0])
            elif len(line) == 4:
                species_n.append(atomic_number[line[0]])

    assert(q_net != None)
    with open('q_net.txt','w') as fp:
        fp.write('%s' %(str(q_net))+'\n')
    species_n = np.array(species_n,dtype=np.intc)
    np.save('traj.npy',coords)
    np.save('species_n.npy',species_n)
    os.chidr(cwd)

# we need to save to traj.hdf5 later, using e3nn image
    #e = np.array(0., dtype=np.single)
    #lst = []
    #for ii in range(len(coords)):
    #    lst.append(dict(pos=coords[ii], energy=e, forces = coords[ii], species=species_n))
    #path = 'traj.hdf5'
    #attrs = {'pos': ('node', '1x1o'), 'species': ('node', '1x0e'), 'energy': ('graph', '1x0e'), 'forces': ('node', '1x1o')}
    #batch = Batch.from_data_list(lst, attrs)
    #batch.dumpHDF5(path)
