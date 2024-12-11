#!/usr/bin/env python

"""
using openbabel to get the config bonded idxs, this information will be used to aviod the unphysical strutrues during the ml md explore 
unreasonable structures must have unresaonable (too long or too short) distance with bonded atoms or too short distance with nonboned atoms
bonded type may also required as we need to know which atoms will be protonated or deprotonated
for protonated fragments, we use rdkit to read the origin topo
if openbabel has probloem, protonated position need to check by yourself
"""
import os
from openbabel import openbabel
from ase import Atoms
from glob import glob
import numpy as np
from scipy.spatial.distance import cdist

vdw_radii = {'H':1.2,'C':1.7,'N':1.55,'O':1.52,'S':1.8}

def q_state(bond_infor):
    cen_atom = bond_infor[0][0]; proton = 0
    nei_ele = sorted([u[1] for u in bond_infor])
    if cen_atom == 'S':
        if len(bond_infor) == 2:
            if bond_infor[0][2] == 1 and bond_infor[1][2] == 1 and sorted(['C','H']) == nei_ele:
                proton = 1
        elif len(bond_infor) == 1:
            if bond_infor[0][1] == 'C' and bond_infor[0][2] == 1:
                proton = -1

    elif cen_atom == 'N':
        if len(bond_infor) == 4:
            if sorted(['C','H','H','H']) == nei_ele:
                proton = 1
        elif len(bond_infor) == 3:
            if bond_infor[0][2] == 1 and bond_infor[1][2] == 1 and bond_infor[2][2] == 1 and sorted(['C','H','H']) == nei_ele:
                proton = -1
            elif bond_infor[0][2] + bond_infor[1][2] + bond_infor[2][2] == 4 and sorted(['C','C','H']) == nei_ele:
                proton = 1
            elif bond_infor[0][2] + bond_infor[1][2] + bond_infor[2][2] == 4 and sorted(['C','H','H']) == nei_ele:
                proton = 1
        elif len(bond_infor) == 2:
            if bond_infor[0][2] + bond_infor[1][2] == 3 and sorted(['C','C']) == nei_ele:
                proton = -1
            elif bond_infor[0][2] + bond_infor[1][2] == 3 and sorted(['C','H']) == nei_ele:
                proton = -1

    elif cen_atom == 'O':
        if len(bond_infor) == 2:
            if bond_infor[0][2] == 1 and bond_infor[1][2] == 1 and sorted(['C','H']) == nei_ele:
                proton = 1
        elif len(bond_infor) == 1:
            if bond_infor[0][2] == 1 and bond_infor[0][1] == 'C':
                proton = -1
    return proton

def _crd2frag(symbols, crds, q_net=None, pbc=False, cell=None):
    # get the bond idxs and bond types
    atomnumber = len(symbols)
    all_atoms = Atoms(symbols = symbols, positions = crds, pbc=pbc, cell=cell)
    mol = openbabel.OBMol(); mol.BeginModify()
    for idx, (num, position) in enumerate(zip(all_atoms.get_atomic_numbers(),all_atoms.positions)):
        atom = mol.NewAtom(idx); atom.SetAtomicNum(int(num))
        atom.SetAtomicNum(int(num)); atom.SetVector(*position)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.EndModify()
    bonds = []
    for ii in range(mol.NumBonds()):
        bond = mol.GetBond(ii)
        a = bond.GetBeginAtom().GetId()
        b = bond.GetEndAtom().GetId()
        bo = bond.GetBondOrder()
        bonds.extend([[a, b, bo], [b, a, bo]])
    bonds = np.array(bonds, ndmin=2).reshape((-1,3))
    atomic_infor = {}
    for ii in range(atomnumber):
        atomic_infor[ii] = []
    for idx,bond in enumerate(bonds):
        atomic_infor[bond[0]].append([symbols[bond[0]],symbols[bond[1]],bond[2]])
    atom_protonated_cand = []
    for ii in range(atomnumber):
        bond_infor = atomic_infor[ii]
        proton = q_state(bond_infor)
        atom_protonated_cand.append(proton)
    protonated_pair = []
    for bond in bonds:
        protonated_pair.append([atom_protonated_cand[bond[0]],atom_protonated_cand[bond[1]]])
    return bonds, protonated_pair

def bond_simply(bonds,coord,symbol):
    # judge whethre the conformer is reasonable quickly
    reason = True
    for bond in bonds:
        b_0 = bond[0]; b_1 = bond[1]
        vdw_0 = vdw_radii[symbol[b_0]]; vdw_1 = vdw_radii[symbol[b_1]]
        c_0 = coord[b_0]; c_1 = coord[b_1]
        dis = np.sqrt(np.mean((c_0 - c_1)**2))
        if dis > (vdw_0 + vdw_1) * 0.7:
            reason = False
    return reason 

def mol_distance(coord):
    # get the mol distance with differnet atoms
    dis_mat = cdist(coord,coord,'euclidean')
    return dis_mat

def proton_transfer_check(coord,symbol,idx_0,idx_1,proton_idx):
    proton_pair = False
    if symbol[idx_0] == 'H' and proton_idx[idx_1] != 0:
        dis = np.sqrt(np.sum((coord - coord[idx_0])**2,axis=1))
        dis_arg = np.argsort(dis)[1:3]
        neg0 = proton_idx[dis_arg[0]]; neg1 = proton_idx[dis_arg[1]]
        if neg0 != 0 and neg1 != 0 and neg0 + neg1 == 0:
            proton_pair = True
        dis = np.sqrt(np.sum((coord - coord[idx_0])**2,axis=1))
        dis_0 = dis[np.argsort(dis)[1]]
    elif symbol[idx_1] == 'H' and proton_idx[idx_0] != 0:
        dis = np.sqrt(np.sum((coord - coord[idx_1])**2,axis=1))
        dis_arg = np.argsort(dis)[1:3]
        neg0 = proton_idx[dis_arg[0]]; neg1 = proton_idx[dis_arg[1]]
        if neg0 != 0 and neg1 != 0 and neg0 + neg1 == 0:
            proton_pair = True
        dis = np.sqrt(np.sum((coord - coord[idx_1])**2,axis=1))
        dis_0 = dis[np.argsort(dis)[1]]
    return proton_pair, dis

def bond_length_static(coord,symbol,bonder):
    bond_length = {}
    for bond in bonder:
        key0 = sorted([symbol[bond[0]],symbol[bond[1]]])
        bond_length[key0] = []
    for bond in bonder:
        key0 = sorted([symbol[bond[0]],symbol[bond[1]]])
        dis = np.sqrt(np.sum((coord[bond[0]] - coord[bond[1]])**2))
        bond_length[key0].append(dis)
    return bond_length

def reasonable_judge(coord,symbol,bonder,proton_idx,bond_hi_all,bond_lo_all):
    # verify whether the configure is reasonable
    dis_mat = mol_distance(coord); reasonable = True
    for i in range(len(coord)):
        for j in range(i+1,len(coord)):
            s = sorted([symbol[i],symbol[j]])
            bond_hi = bond_hi_all[s]; bond_lo = bond_hi_all[s]
            atom_dis = dis_mat[i,j]
            # check whether proton pair
            proton_pair = False
            if proton_idx[i] != 0 or proton_idx[j] != 0:
                proton_pair,dis_hydro = proton_transfer_check(coord,symbol,i,j,proton_idx)

            # check whether bonded atoms has too large or small distance 
            if [i,j] in bonder:
                if proton_pair == True:
                    # may be hydrogen is transfer to another heavy atoms
                    if atom_dis < bond_lo or dis_hydro > bond_hi:
                        reasonable = False
                else:
                    if atom_dis < bond_lo or atom_dis > bond_hi:
                        reasonable = False
            # check whether nonbonded atoms has too small distance
            else:
                if proton_pair == True:
                    if atom_ids < bond_lo * 0.9:
                        reasonable = False
                else:
                    if atom_dis < (bond_hi + bond_lo)/2.:
                        reasonable = False
    return reasonable

if __name__ == '__main__':
    """
    generate the topology informations for configurations
    """
    dirs = glob('./poscar/*/*/*single.log')
    ele_dict = ['X','C','H','N','O','S']
    coord_conf = []; symbol_conf = []; qnet_conf = []
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
    
    cwd = os.getcwd()
    for dir0 in dirs:
        f_name = os.path.basename(dir0)
        dir_name = dir0[:-len(f_name)]
        os.chdir(dir_name)
        coord, symbol = geometry_read(f_name)
        bond, proton = _crd2frag(symbol, coord)
        with open('topo.txt','w') as fp:
            fp.write('bond_idx_0; bond_idx_1; bond_type; proton_idx_0; proton_idx_1'+'\n')
            for ii,jj in zip(bond,proton):
                fp.write('%s %s %s %s %s'%(str(ii[0]),str(ii[1]),str(ii[2]),str(jj[0]),str(jj[1]))+'\n')
        os.chdir(cwd)

    #bond_idxs = []
    #for ii,cc in enumerate(coord_conf):
    #    bond, proton = _crd2frag(symbol_conf[ii], cc, qnet)
    #    bond_idxs.append(bond)

