#!/usr/bin/env python
"""
generate dimer config, first we find the farthest two atoms and the mass center; then according to the direction 
move the monomer, find the smallest distance between the monomer. If the distance arrive the threshold, then stop
here we only consider the A-A type, for A-B type, just pull the mass center of two monomers
"""
import os
import numpy as np
import ase
from scipy.spatial.distance import cdist
import copy
from glob import glob

# shift along the center of mass direction
def find_closest_distance(coord_A, coord_B):
    n_atoms1 = len(coord_A); n_atoms2 = len(coord_B)
    min_i = -1; min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = np.array(coord_A[i])
        for j in range(n_atoms2):
            r2 = np.array(coord_B[j])
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

def move_monomer(coord_A,symbol_A,coord_B,symbol_B,bond_infor,r_max,direction,dr,mono_A_n):
    bond_infor_n = []
    for bond in bond_infor:
        bond_n = [bond[0] + mono_A_n, bond[1] + mono_A_n, bond[2], bond[3], bond[4], bond[5]]
        bond_infor_n.append(bond_n)
    _,_,min_dir = find_closest_distance(coord_A, coord_B)
    while min_dir < r_max:
        coord_B += direction * dr 
        _,_,min_dir = find_closest_distance(coord_A, coord_B)
    return coord_B, symbol_B, bond_infor_n

def move_direction(coord,symbol):
    masses = ase.data.atomic_masses
    atomic_numbers = ase.data.atomic_numbers
    symbol = np.array([masses[atomic_numbers[u]] for u in symbol]).reshape(-1,1)
    mass_cen = np.sum(coord * symbol,axis=0)
    dis_mat = cdist(coord,coord,'euclidean')
    idx_max = np.argmax(dis_mat)
    idx_0 = idx_max // len(dis_mat); idx_1 = idx_max % len(dis_mat)     
    vec_0 = coord[idx_0] - mass_cen; vec_1 = coord[idx_1] - mass_cen
    vec_dir = np.cross(vec_0,vec_1)
    if np.sum(np.abs(vec_dir)) < 0.1:
        vec_dir = np.array([0.,0.,1.])
    vec_dir /= np.linalg.norm(vec_dir)
    return vec_dir

def data_read(f_inp,f_top):
    coord = []; symbol = []; proton = None
    with open(f_inp,'r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 4:
                coord.append([float(line[1]),float(line[2]),float(line[3])])
                symbol.append(line[0])
            if len(line) == 2:
                if int(line[-1]) == 1:
                    proton = int(line[0])
    bond_infor = []
    with open(f_top,'r') as fp:
        for idx,line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                bond_tmp = [int(line[0]),int(line[1]),\
                        int(line[2]),int(line[3]),int(line[4]),\
                        float(line[5])]
                bond_infor.append(bond_tmp)
    return coord, symbol, bond_infor, proton 

def data_write(coord_A, symbol_A, coord_B, symbol_B, bond_infor_A, bond_infor_B, proton_A, proton_B):
    #with open('dimer_scan.gjf','w') as fp:
    with open('dimer_sapt.gjf','w') as fp:
        fp.write('%nproc=8'+'\n')
        fp.write('#wb97xd/6-31g* force'+'\n')
        fp.write('\n'); fp.write('DPGEN'+'\n')
        fp.write('\n'); fp.write('%s 1' %(str(proton_A+proton_B))+'\n')
        for ss,cc in zip(symbol_A,coord_A):
            fp.write('%s %.5f %.5f %.5f' %(ss,cc[0],cc[1],cc[2])+'\n')
        for ss,cc in zip(symbol_B,coord_B):        
            fp.write('%s %.5f %.5f %.5f' %(ss,cc[0],cc[1],cc[2])+'\n')
        fp.write('\n')
    with open('dimer_topo.txt','w') as fp:
         fp.write('bond_idx_0; bond_idx_1; bond_type; proton_idx_0; proton_idx_1; bond_length'+'\n')
         for ii in bond_infor_A:
             fp.write('%s %s %s %s %s %s' %(str(ii[0]),str(ii[1]),str(ii[2]),str(ii[3]),str(ii[4]),str(ii[5]))+'\n')
         for ii in bond_infor_B:
             fp.write('%s %s %s %s %s %s' %(str(ii[0]),str(ii[1]),str(ii[2]),str(ii[3]),str(ii[4]),str(ii[5]))+'\n')
    with open('dimer_num.txt','w') as fp:
        fp.write('%s' %(str(len(coord_A)))+'\n'); fp.write('%s' %(str(len(coord_B)))+'\n')
    return 

if __name__ == '__main__':
    # r_max = 3 is dimer config; r_max = 5 used for sapt 
    r_max = 5; dr = 0.2
    f_dirs = glob('./*/*/*/*single.gjf'); cwd_ = os.getcwd()
    for dir0 in f_dirs:
        print('*********************')
        print(dir0)
        f_name = os.path.basename(dir0)
        f_dir = dir0[:-len(f_name)]
        os.chdir(f_dir)
        coord, symbol, bond_infor, proton = data_read(f_name,'topo.txt')
        direction = move_direction(coord, symbol)
        coord_n, symbol_n, bond_infor_n = move_monomer(coord, symbol, coord, symbol, bond_infor, r_max, direction, dr, len(coord))
        data_write(coord,symbol,coord_n,symbol_n,bond_infor,bond_infor_n,proton,proton)
        os.chdir(cwd_) 
