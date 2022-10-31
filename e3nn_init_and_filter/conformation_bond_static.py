#!/usr/bin/env python
"""
static different bond length range from fp data, we need to know how the atom order changed from conf to lammps
"""
import numpy as np
import numpy as np
from glob import glob 
import os
import json
from organic_topo import _crd2frag
from scipy.spatial.distance import cdist

def geometry_read(f_name):
    read_c = False; coord = []; symbol = []
    with open(f_name) as fp:
        for line in fp:
            if 'Charge' in line and 'Multiplicity' in line:
                read_c = True
            elif read_c == True:
                line1 = line.strip().split()
                if len(line1) == 4:
                    coord.append([float(line1[1]),float(line1[2]),float(line1[3])])
                    symbol.append(line1[0])
            if 'GradGradGrad' in line:
                read_c = False
                break
    return coord, symbol


def bond_dis(coord_mat,symbol,bonds,bonds_dis_dict,proton_pair):
    unnorm = False
    for bond in bonds:
        sym = tuple(sorted([symbol[bond[0]],symbol[bond[1]]]))
        dis = coord_mat[bond[0],bond[1]]
        #if dis > 15:
        #    unnorm = True
        bonds_dis_dict[sym].append(dis)
        if dis > 4:
            if proton_pair[bond[0]] == 0 and proton_pair[bond[1]] == 0:
                unnorm = True
        else:
            bonds_dis_dict[sym].append(dis)
    return bonds_dis_dict, unnorm

def convert_bond(bond,symbol,element):
    # as dpdata transfer the data will change the atoms order, so we need to obatain the changed bond order
    atom_ord = np.arange(len(symbol)); k = 0
    for ele in elements:
        for ii,sym in enumerate(symbol):
            if sym == ele:
                atom_ord[ii] = k 
                k += 1
    bond_final = []
    for ii in bond:
        bond_final.append([atom_ord[ii[0]],atom_ord[ii[1]],ii[2]])
    return bond_final

if __name__ == '__main__':
    with open('dpgen_1_param.json','r') as fp:
        parm = json.load(fp)

    sys_configs = parm['sys_coinfigs']

    # get different config's bond connection
    bonds = []; elements = ['C','H','N','O','S']; charge_pairs = []
    for ii,sys in enumerate(sys_configs):
        f_name = './poscar/'+sys[0]
        f_name1 = os.path.basename(f_name)
        f_dir = f_name[:-len(f_name1)]+'topo.txt'
        bond = []; charge_pair = []
        with open(f_dir,'r') as fp:
            for idx, line in enumerate(fp):
                if idx > 0:
                    line1 = line.strip().split()
                    bond.append([int(line1[0]),int(line1[1]),int(line1[2])])
                    charge_pair.append([int(line1[3]),int(line1[4])])
        #coord, symbol = geometry_read(f_name)
        #bond, proton_pair = _crd2frag(symbol, coord, 0)
        #bond = convert_bond(bond,symbol,elements)
        bonds.append(bond); charge_pairs.append(charge_pair)
    
    bonds_dis_dict = {}
    for ele0 in elements:
        for ele1 in elements:
            sym = tuple(sorted([ele0,ele1]))
            bonds_dis_dict[sym] = []
    
    # get bonded atoms distance
    dirs = glob('./iter.00001*/02.fp/data.*')
    type_map = ['C','H','N','O','S']
    for dir0 in dirs:
        f_name = os.path.basename(dir0)
        ii = int(f_name.split('.')[-1])
        coord = np.load(dir0+'/set.000/coord.npy')
        symbol = np.loadtxt(dir0+'/type.raw')
        symbol = [type_map[int(u)] for u in symbol]
        for idx,cc in enumerate(coord):
            cc = cc.reshape((int(len(cc)/3),3))
            coord_mat = cdist(cc,cc,'euclidean')
            bonds_dis_dict,unnorm = bond_dis(coord_mat,symbol,bonds[ii],bonds_dis_dict,charge_pairs[ii])
            if unnorm == True:
                print(dir0)
                print(idx)
    print(bonds_dis_dict.keys())
    a = (bonds_dis_dict[('C','C')])
    print(np.max(a))
    print(np.mean(a))
    print(np.min(a))
