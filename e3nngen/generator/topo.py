#!/usr/bin/env python
"""
verify whether the conformation has the similar topo with the origin config during dpgen progress
convert lammpstrj file to dp file
"""
import numpy as np
import sys
from scipy.spatial.distance import cdist
from glob import glob
import dpdata

bond_hi_ratio = float(sys.argv[1])
bond_lo_ratio = float(sys.argv[2])
trj_freq = int(sys.argv[3])

def topo_bond(num_atom):
    bonds = []; atom_pair_lengths = {}; proton_idx = [0]*num_atom
    bonds_lengths = {}
    with open('conf.topo','r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                bonds.append([int(line[0]),int(line[1])])
                bonds_lengths[tuple([int(line[0]),int(line[1])])] = (float(line[5]))
                proton_idx[int(line[0])] = int(line[3])
                proton_idx[int(line[1])] = int(line[4])
    
    with open('conf.bondlength','r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                sym = tuple(sorted([line[0],line[1]]))
                atom_pair_lengths[sym] = float(line[2])

    return bonds, atom_pair_lengths, proton_idx, bonds_lengths

def mol_distance(coord):
    dis_mat = cdist(coord,coord,'euclidean')
    return dis_mat

def proton_transfer_check0(coord,symbol,idx_hydro,idx_0,proton_idx,bond_lengths):
    # if the origin bond pair is not bonded, check whether the charge transfer is occurred
    proton_pair = False
    c_hy = coord[idx_hydro]
    dis = np.sqrt(np.sum((coord - c_hy)**2,axis=1))
    dis_ord = np.argsort(dis)[1]
    if dis_ord != idx_0 and proton_idx[dis_ord] < 0 and proton_idx[idx_0] > 0:
        sym = tuple(sorted([symbol[dis_ord],'H'])) 
        if dis[dis_ord] > bond_lengths[sym] * bond_lo_ratio and dis[dis_ord] < bond_lengths[sym] * bond_hi_ratio:
            proton_pair = True
    return proton_pair

def proton_transfer_check1(coord,symbol,idx_hydro,idx_0,proton_idx,bond_lengths):
    # if the origin nonboned pair seems bonded, check whether the charge transfer is occurred
    proton_pair = False
    c_hy = coord[idx_hydro]
    dis = np.sqrt(np.sum((coord[idx_0] - c_hy)**2))
    if proton_idx[idx_0] < 0:
        sym = tuple(sorted([symbol[idx_0],'H']))
        if dis > bond_lengths[sym] * bond_lo_ratio and dis < bond_lengths[sym] * bond_hi_ratio:
            proton_pair = True
    return proton_pair

def reasonable_judge(coord,symbol,bonds,atom_pair_lengths,proton_idx,bonds_lengths):
    # verify whether the configure is reasonable
    coord = np.array(coord)
    dis_mat = mol_distance(coord); reasonable = True; cri_loose = 0
    for i in range(len(coord)):
        for j in range(i+1,len(coord)):
            atom_dis = dis_mat[i,j]
            if [i,j] in bonds:
                # bonded pair
                bond_length = bonds_lengths[tuple([i,j])]
                bond_hi = bond_length * bond_hi_ratio; bond_lo = bond_length * bond_lo_ratio
                if atom_dis > bond_lo and atom_dis < bond_hi:
                    pass
                elif atom_dis < bond_lo:
                    reasonable = False
                else:
                    if symbol[i] == 'H':
                        reasonable = proton_transfer_check0(coord,symbol,i,j,proton_idx,atom_pair_lengths)
                    elif symbol[j] == 'H':
                        reasonable = proton_transfer_check0(coord,symbol,j,i,proton_idx,atom_pair_lengths)
                    else:
                        reasonable = False
                # the criteration for S should be loose; 0 means reasonable or no S; 1 means unreasonable caused by S; 
                # 2 means unreasonable caused by other atoms
                if reasonable == False:
                    if ((symbol[i] == 'S' and symbol[j] != 'H') or (symbol[i] != 'H' and symbol[j] == 'S')) and cri_loose < 2:
                        cri_loose = 1
                    else:
                        cri_loose = 2
            else:
                # nonbonded pair
                sym = tuple(sorted([symbol[i],symbol[j]]))
                # need to define after consider
                bond_hi = atom_pair_lengths[sym] * (bond_hi_ratio - 0.3)
                bond_lo = atom_pair_lengths[sym] * bond_lo_ratio
                if atom_dis > bond_hi:
                    pass
                elif atom_dis < bond_lo:
                    reasonable = False
                else:
                    if symbol[i] == 'H':
                        reasonable = proton_transfer_check1(coord,symbol,i,j,proton_idx,atom_pair_lengths)      
                    elif symbol[j] == 'H':
                        reasonable = proton_transfer_check1(coord,symbol,j,i,proton_idx,atom_pair_lengths)
                    else:
                        reasonable = False
                if reasonable == False:
                    if ((symbol[i] == 'S' and symbol[j] != 'H') or (symbol[i] != 'H' and symbol[j] == 'S')) and cri_loose < 2:
                        cri_loose = 1
                    else:
                        cri_loose = 2 
    return reasonable,cri_loose

def lammpsread(f_name):
    coord = []
    with open(f_name,'r') as fp:
        read_coord = False
        for line in fp:
            if 'ATOMS' in line and 'type' in line and 'id' in line:
                read_coord = True
            elif read_coord == True:
                line1 = line.strip().split()
                coord.append([float(line1[2]),float(line1[3]),float(line1[4])])
    return coord

type_map_0 = ['X','C','H','N','O','S']; symbol = []
with open('conf.lmp','r') as fp:
    read_sym = False
    for line in fp:
        if 'Atoms' in line and 'atomic' in line:
            read_sym = True
        elif read_sym == True:
            line1 = line.strip().split()
            if len(line1) == 5:
                symbol.append(type_map_0[int(line1[1])])

bonds, atom_pair_lengths, proton_idx, bonds_lengths = topo_bond(len(symbol))
n_lammps_files = len(glob('./traj/*.lammpstrj'))

reasons = []
with open('reasonable.txt','w') as fp:
    for ii in range(n_lammps_files):
        f_name = str(ii * trj_freq)+'.lammpstrj'
        coord = lammpsread('./traj/'+f_name)
        reasonable, cri_loose = reasonable_judge(coord,symbol,bonds,atom_pair_lengths,proton_idx,bonds_lengths)
        if reasonable == True:
            reasons.append(1)
            # reasonable
            fp.write('1 1'+'\n')
        else:
            reasons.append(0)
            if cri_loose == 1:
                # loose reasonable
                fp.write('0 1'+'\n')
            else:
                # unreasonable
                fp.write('0 0'+'\n')

for ii in range(n_lammps_files):
    f_name = str(ii * trj_freq)+'.lammpstrj'
    if ii == 0:
        system = dpdata.System('./traj/'+str(f_name),fmt='dump',type_map=type_map_0[1:])
    else:
        if reasons[ii] == 1:
            system.append(dpdata.System('./traj/'+str(f_name),fmt='dump',type_map=type_map_0[1:]))
        else:
            system.append(dpdata.System('./traj/0.lammpstrj',fmt='dump',type_map=type_map_0[1:]))

system.nopbc = True
system.to_deepmd_npy('traj_deepmd')
