#!/usr/bin/env python
"""
obatin the different bond length of config and static the mean length of bonds for each config
"""
from glob import glob
from scipy.spatial.distance import cdist
import numpy as np
import os

def topo_read(f_name):
    topo = [] 
    with open(f_name,'r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                topo.append([int(u) for u in line[0:5]])
    return topo

def topo_bond_dis(topo,dis_mat,symbol):
    bond_mean = {}; topo1 = []
    
    for bond in topo:
        sym = sorted([symbol[bond[0]],symbol[bond[1]]])
        bond_mean[tuple(sym)] = []
    
    for idx,bond in enumerate(topo):
        dis = dis_mat[bond[0],bond[1]]
        topo1.append(topo[idx] + [dis])
        sym = sorted([symbol[bond[0]],symbol[bond[1]]])
        bond_mean[tuple(sym)].append(dis)

    for key,value in bond_mean.items():
        bond_mean[key] = np.mean(bond_mean[key])

    return topo1, bond_mean

def topo_write(topo,bond_mean):
    with open('topo.txt','w') as fp:
        fp.write('bond_idx_0; bond_idx_1; bond_type; proton_idx_0; proton_idx_1; bond_length'+'\n')
        for bond in topo:
            for ii in range(len(bond)):
                if ii == len(bond) - 1:
                    fp.write('%s' %(str(bond[ii]))+'\n')
                else:
                    fp.write('%s' %(str(bond[ii]))+'\t')
    with open('atompair_length.txt','w') as fp:
        fp.write('type_idx_0; type_idx_1; bond_length'+'\n')
        for key, value in bond_mean.items():
            fp.write('%s %s %s' %(str(key[0]),str(key[1]),str(value))+'\n')
    return 

if __name__ == '__main__':
    #dirs = glob('./poscar/*/*/*single.log')
    dirs = glob('./poscar/*/*/*/*single.log')
    cwd = os.getcwd()
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
 
    for dir0 in dirs:
        f_name = os.path.basename(dir0) 
        f_dir = dir0[:-len(f_name)]
        os.chdir(f_dir)
        topo = topo_read('topo.txt')
        coord,symbol = geometry_read(f_name)
        coord_mat = cdist(coord,coord,'euclidean')
        topo, bond_mean = topo_bond_dis(topo,coord_mat,symbol)
        topo_write(topo, bond_mean)    
        os.chdir(cwd) 
    bond_mean = {}
    for ii in ['C','H','O','N','S']:
        for jj in ['C','H','O','N','S']:
            sym = sorted([ii,jj])
            bond_mean[tuple(sym)] = []
    cwd = os.getcwd()
    for dir0 in dirs:
        f_name = os.path.basename(dir0)
        f_dir = dir0[:-len(f_name)]
        os.chdir(f_dir)
        with open('atompair_length.txt','r') as fp:
            for idx,line in enumerate(fp):
                if idx > 0:
                    line = line.strip().split()
                    sym = tuple(sorted([line[0],line[1]]))
                    bond_mean[sym].append(float(line[2]))
        os.chdir(cwd)
    #with open('atompair_length.txt','w') as fp:
    #    fp.write('type_idx_0; type_idx_1; bond_length'+'\n')
    #    for key, value in bond_mean.items():
    #        if len(value) > 0:
    #            fp.write('%s %s %s' %(str(key[0]),str(key[1]),str(np.mean(np.array(value))))+'\n')
    #        else:
    #            fp.write('%s %s %s' %(str(key[0]),str(key[1]),str(0.))+'\n')
