#!/usr/bin/env python
"""
filt the unreasonable conformations according the configuration topo
"""
import sys
import numpy as np
import sys
from scipy.spatial.distance import cdist
from glob import glob
from monty.serialization import loadfn
import os

bond_hi_ratio = float(sys.argv[1])
bond_lo_ratio = float(sys.argv[2])

def topo_bond(num_atom):
    bonds = [];  proton_idx = [0]*num_atom
    bonds_lengths = {}
    with open('topo.txt','r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                bonds.append([int(line[0]),int(line[1])])
                bonds_lengths[tuple([int(line[0]),int(line[1])])] = (float(line[5]))
                proton_idx[int(line[0])] = int(line[3])
                proton_idx[int(line[1])] = int(line[4])

    return bonds, proton_idx, bonds_lengths

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
    strange_ratio = []
    dis_mat = mol_distance(coord); reasonable = True
    for i in range(len(coord)):
        for j in range(i+1,len(coord)):
            atom_dis = dis_mat[i,j]
            if [i,j] in bonds:
                # bonded pair
                bond_length = bonds_lengths[tuple([i,j])]
                bond_hi = bond_length * bond_hi_ratio; bond_lo = bond_length * bond_lo_ratio
                strange_ratio.append(atom_dis/bond_length)
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
            else:
                # nonbonded pair
                sym = tuple(sorted([symbol[i],symbol[j]]))
                # need to define after consider
                bond_hi = atom_pair_lengths[sym] * (bond_hi_ratio-0.2)
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
    strange_lo = np.min(strange_ratio); strange_hi = np.max(strange_ratio)
    return reasonable, strange_lo, strange_hi

def bond_transfer(bonds, proton_idx, bonds_lengths, symbol, type_map):
    _idx_transfer = []
    for ii in type_map:
        for jj in range(len(symbol)):
            if symbol[jj] == ii:
                _idx_transfer.append(jj)
    idx_transfer = {}
    for idx, ii in enumerate(_idx_transfer):
        idx_transfer[ii] = idx

    _bonds = []; _proton_idx = [0] * len(proton_idx); _bonds_lengths = {}
    
    for bond in bonds:
        _bonds.append([idx_transfer[bond[0]],idx_transfer[bond[1]]])
        key_ori = tuple([int(bond[0]),int(bond[1])])
        key_new = tuple([idx_transfer[bond[0]],idx_transfer[bond[1]]])
        _bonds_lengths[key_new] = bonds_lengths[key_ori]
        _proton_idx[idx_transfer[bond[0]]] = proton_idx[bond[0]]
        _proton_idx[idx_transfer[bond[1]]] = proton_idx[bond[1]]
    return _bonds, _proton_idx, _bonds_lengths

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

def gaussianread(f_name):
    symbol = []
    with open(f_name,'r') as fp:
        read_symbol = False
        for line in fp:
            if line.startswith(' Symbolic Z-matrix:') and len(symbol) == 0:
                read_symbol = True
            if read_symbol == True:
                line1 = line.strip().split()
                if len(line1) == 4:
                    symbol.append(line1[0])
            if line.startswith(' GradGradGradGradGradGradGrad'):
                read_symbol = False
    return symbol

#======================================================================================================
# read config informations; need param.json to know the order of configs
jdata = loadfn(sys.argv[3])
sys_configs_prefix = jdata['sys_configs_prefix']
sys_configs = jdata['sys_configs']
_sys_configs = []
for ss in sys_configs:
    _sys_configs.append(os.path.join(sys_configs_prefix,ss[0]))

#print(_sys_configs[168])
pair_lengths = {}
with open('atompair_length.txt','r') as fp:
    for idx, line in enumerate(fp):
        if idx > 0:
            line = line.strip().split()
            sym = tuple(sorted([line[0],line[1]]))
            pair_lengths[sym] = float(line[2])

bonds = []; proton_idxs = []; bonds_lengths = []; config_symbols = []

cwd_ = os.getcwd()
for dir0 in _sys_configs:
    dir_name = dir0[:-len(os.path.basename(dir0))]
    os.chdir(dir_name)
    f_name = glob('./*single.log')
    assert(len(f_name) == 1)
    symbol = gaussianread(f_name[0])
    bond, proton_idx, bond_length = topo_bond(len(symbol))
    bonds.append(bond); proton_idxs.append(proton_idx)
    bonds_lengths.append(bond_length); config_symbols.append(symbol)
    os.chdir(cwd_)
#=======================================================================================================

fp_dirs = glob('./iter.*/02.fp/data.*')
type_map = ['C','H','N','O','S']
type_map_0 = ['X','C','H','N','O','S']

cwd_ = os.getcwd(); break_bond_hi = []; break_bond_lo = []
for dir0 in fp_dirs:
    conf_idx = int(os.path.basename(dir0).split('.')[1])
    bond = bonds[conf_idx]; proton_idx = proton_idxs[conf_idx]
    bond_length = bonds_lengths[conf_idx]; config_symbol = config_symbols[conf_idx]
    _bond, _proton_idx, _bond_length = bond_transfer(bond, proton_idx, bond_length, config_symbol, type_map)
    os.chdir(dir0)
    coord = np.load(os.path.join('set.000','coord.npy'))
    reasons = []; stranges_lo = []; stranges_hi = []
    type_ele = np.loadtxt('type.raw',dtype=int)
    symbol = [type_map[u] for u in type_ele]

    for cc in coord:
        cc = cc.reshape((int(len(cc)/3),3))
        reason,s_lo,s_hi = reasonable_judge(cc,symbol,_bond,pair_lengths,_proton_idx,_bond_length)
        reasons.append(reason); stranges_hi.append(s_hi); stranges_lo.append(s_lo)
    # collect whether reason, the longest and shortest bond ratio
    break_bond_hi.append(np.max(stranges_hi)); break_bond_lo.append(np.min(stranges_lo))
    with open('reason.txt','w') as fp:
        for idx,reason in enumerate(reasons):
            fp.write('%s %s' %(str(stranges_hi[idx]),str(stranges_lo[idx]))+'\t')
            if reason == True:
                fp.write('1'+'\n')
            else:
                fp.write('0'+'\n')
                print(dir0)
    os.chdir(cwd_)

print(np.max(break_bond_hi))
print(np.min(break_bond_lo))
