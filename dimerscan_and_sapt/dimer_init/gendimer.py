#!/usr/bin/env python
"""
generate dimer config, first we find the farthest two atoms and the mass center; then according to the direction 
move the monomer, find the smallest distance between the monomer. If the distance arrive the threshold, then stop
"""
import numpy as np
import ase
from scipy.spatial.distance import cdit
import copy
# shift along the center of mass direction
def find_closest_distance(coord_A, coord_B):
    n_atoms1 = len(coord_A); n_atoms2 = len(coord_B)
    min_i = -1; min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = pos1[i]
        for j in range(n_atoms2):
            r2 = pos2[j]
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

def move_monomer(coord,symbol,bond_infor,r_max,direction,dr):
    bond_infor_n = []
    for bond in bond_infor:
        bond_n = [bond[0] + len(symbol), bond[1] + len(symbol), bond[2], bond[3], bond[4], bond[5]]
        bond_infor_n.append(bond_n)
    coord_0 = copy.deepcopy(coord)
    _,_,min_dir = find_closest_distance(coord_0, coord)
    while min_dir < r_max:
        coord += direction * dr 
        _,_,min_dir = find_closest_distance(coord_0, coord)
    return coord; symbol, bond_infor_n

def more_direction(coord,symbol):
    masses = ase.data.atomic_masses
    symbol = np.array([masses[u] for u in symbol]).reshape(-1,1)
    mass_cen = np.sum(coord * symbol,axis=0)
    dis_mat = cdist(coord,coord,'euclidean')
    dis_mat_0 = np.max(dis_max,axis=0)
    dis_mat_1 = np.max(dis_max,aixs=1)
    idx_0 = np.argsort(dis_mat_0)[-1]; idx_1 = np.argsort(dis_mat_1)[-1]
    vec_0 = coord[idx_0] - mass_cen; vec_1 = coord[idx_1] - mass_cen
    vec_dir = np.cross(vec_0,vec_1)
    if np.sum(np.abs(vec_dir)) < 0.1:
        vec_dir = [0,0,1]
    return vec_dir

def data_read(f_inp,f_top):
    coord = []; symbol = []
    with open(f_inp,'r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 4:
                coord.append([float(line[1]),float(line[2]),float(line[3])])
                symbol.append(line[0])
    bond_infor = []
    with open(f_top,'r') as fp:
        for idx,line in enumerate(fp):
            if idx > 0:
                line = line.strip.split(); bond_tmp = []
                for uu in line:
                    bond_tmp.append([int(line[0]),int(line[1]),\
                        int(line[2]),int(line[3]),int(line[4]),\
                        float(line[5])])
                bond_infor.append(bond_tmp)
    return coord, symbol, bond_infor

if __name__ == '__main__':
    a = 1
