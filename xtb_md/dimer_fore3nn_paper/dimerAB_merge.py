#!/usr/bin/env python
"""
this script is used to merge the data of dimer AB, all data in confA_B/*/gjfs/*/*/input.gjf and confA_B/*/gjfs/*/*/coord.npy should merge in one file 
"""
import os
from glob import glob 
import numpy as np

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
    # here we can save coord as npy or as e3nn data file
    os.chidr(cwd)
