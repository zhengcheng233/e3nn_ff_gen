#!/usr/bin/env python
"""
this script is used to sample fragments (all the atoms within a distance cutoff) from large molecule or system
first find all the atoms with a distance rc; then find connected or disconnected frag; finally find the nearest atoms 
and generate frag
need further improve: judge the bond order by ourself, using the conn infor 
"""
import numpy as np
from openbabel import openbabel
import MDAnalysis 
from ase import Atoms
from scipy.spatial.distance import cdist

def generate_covalent_map(bond):
    n_max = np.max(bond) + 1
    covalent_map = np.zeros((n_max,n_max),dtype=np.int32)
    atom_idx = {}
    for i in range(n_max):
        atom_idx[i] = []
    for pair in bond:
        atom_idx[pair[0]].append(pair[1])
    for i in range(n_max):
        atom_idx[i] = list(set(atom_idx[i]))
    for i in range(n_max):
        visited = [i]
        for j in atom_idx[i]:
            covalent_map[i,j] = 1
            visited.append(j)
            for k in atom_idx[j]:
                if k not in visited:
                    covalent_map[i,k] = 2
                    visited.append(k)
                else:
                    continue
                for l in atom_idx[k]:
                    if l not in visited:
                        covalent_map[i,l] = 3
                        visited.append(l)
                    else:
                        continue
                    for m in atom_idx[l]:
                        if m not in visited:
                            covalent_map[i,m] = 4
                            visited.append(m)
                        else:
                            continue
                        for u in atom_idx[m]:
                            if u not in visited:
                                covalent_map[i,u] = 5
                                visited.append(u)
                            else:
                                continue
                            for v in atom_idx[u]:
                                if v not in visited:
                                    covalent_map[i,v] = 6
                                    visited.append(v)
                                else:
                                    continue
    return covalent_map

def _crd2frag(symbols, crds, pbc=False, cell=None):
    # get the bond idxs and bond types
    atomnumber = len(symbols)
    all_atoms = Atoms(symbols = symbols, positions = crds, pbc=False, cell=None)
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
    return bonds

def _bonder(bonds,symbols):
    n_orb = {'H':1,'C':4,'N':3,'O':2,'S':2}
    _bonds = []
    for bond in bonds:
        _bonds.append(bond); _bonds.append([bond[1],bond[0]])
    n_conn = np.zeros(np.max(_bonds)+1,dtype=np.int32)
    for bond in _bonds:
        n_conn[bond[0]] += 1
    n_ord = np.ones(np.max(_bonds)+1,dtype=np.int32)
    for ii in range(len(symbols)):
        n_ord[ii] = n_ord[ii] + n_orb[symbols[ii]] - n_conn[ii]
    bonds_ord = []
    for bond in _bonds:
        bond_tmp = [bond[0],bond[1]]
        if n_ord[bond[0]] > 1 and n_ord[bond[1]] > 1:
            assert(n_ord[bond[0]] == n_ord[bond[1]])
            bond_tmp.append(n_ord[bond[0]])
        else:
            bond_tmp.append(1)
        bonds_ord.append(bond_tmp)
    return bonds_ord

#def bond_merge():
"""
maybe needed if molecule is too large
"""


def frag_gen(ligs, cov_idx, symbol, n_heavy=8):
    def _frag_gen(cen_idx, frg_idx, cov_idx, symbol):
        n_atom = 0; frg_idx_fin = [cen_idx]
        for i in range(1,7):
            nei_idx = np.where(cov_idx[cen_idx] == i)[0]
            for j in nei_idx:
                if n_atom >= n_heavy:
                    continue
                else:
                    if j in frg_idx:
                        if symbol[j] != 'H':
                            n_atom += 1
                        frg_idx_fin.append(j)
        return frg_idx_fin
    
    _ligs = []
    for lig in ligs:
        _lig = _frag_gen(lig[0],lig,cov_idx,symbol,n_heavy)
        _ligs.append(_lig) 
    return _ligs

def cluster_gen(frgs, cov_idx, bond_type, symbol):
    """ 
    add hydrogen atoms or heavy atoms for bond saturation 
    """
    _frgs = []
    for frg in frgs:
        _frgs.extend(frg)
    _frgs = list(set(_frgs))
    
    # get the bond_type
    _bond_type = {}
    for type0 in bond_type:
        _bond_type[(type0[0],type0[1])] = type0[2]

    # add atoms; _frgs record the real atoms
    _hydro_pair = []; bounder_atom = [] 
    for ii in _frgs:
        nei = np.where(cov_idx[ii] == 1)[0]
        for jj in nei:
            # whether in frag
            if jj in _frgs:
                pass
            else:
                bounder_atom.append(ii)
    bounder_atom = list(set(bounder_atom))
    nonbounder_atom = list(set(_frgs) - set(bounder_atom))

    # add the hydrogen or heavy atom now
    _frgs_filt = bounder_atom; _pair = []
    #for atom in nonbounder_atom:
        # if all conneted is single bond

        # if only one double bond 

        # if two doublue bond


    # avoid the ring atoms i and j are too near 
    #_frgs_filt = nonbounder_atom; 
    #for ii in range(len(bounder_atom)):
    #    for jj in range(i+1, len(bounder_atom)):
    #        idx0 = bounder_atom[ii]; idx1 = bounder_atom[jj]
    #        nei0 = np.where(cov_idx[idx0] == 1)[0]; nei1 = np.where(cov_idx[idx1] == 1)[0]
    #        if ( len(set(nei0) & set(nei1)) > 0):
    #            pass
    #        else:
    #            _frgs_filt.append(idx0)

    return 

def frag_gen(cen_idx, frg_idx, cov_idx, coord, symbol, bond_type, n_heavy=8):
    """
    determine the size of the fragments
    """ 
    def add_hydron(c0,c1):
        c_h = (c1 - c0)*1.05/np.linalg.norm(c1-c0) + c0 
        return c_h
    _bond_type = {}
    for type0 in bond_type:
        _bond_type[(type0[0],type0[1])] = type0[2]
    frg_idx_fin = [cen_idx]; n_atom = 0 if symbol[cen_idx] == 'H' else 1
    hydro_idx = []
    # obtain the proper size frag
    # !!! the cov_idx have problem, thus may only a vector but now is matrix
    
    for i in range(1,7):
        nei_idx = np.where(cov_idx[cen_idx] == i)[0]
        for j in nei_idx:
            if n_atom >= n_heavy:
                continue
            else:
                if j in frg_idx:
                    if symbol[j] != 'H':
                        n_atom += 1
                    frg_idx_fin.append(j)

    # add atoms in frg_idx_fin for bond saturation
    # !!!! have problem, we should add after merge all teh frag
    for ii in frg_idx_fin:
        nei = np.where(cov_idx[ii] == 1)[0]
        for jj in nei:
            if jj not in frg_idx_fin:
                if _bond_type[(ii,jj)] > 1:
                    frg_idx_fin.append(jj)
                    nei = np.where(cov_idx[jj] == 1)[0]
                    for kk in nei:
                        if kk not in frg_idx_fin:
                            hydro_idx.append([jj,kk])
                else:
                    hydro_idx.append([ii,jj])

    coord_fin = [coord[u] for u in frg_idx_fin]
    symbol_fin = [symbol[u] for u in frg_idx_fin]
    for ii in hydro_idx:
        symbol_fin.append('H')
        h_coord = add_hydron(coord[ii[0]],coord[ii[1]])
        coord_fin.append(h_coord)
    return frg_idx_fin, hydro_idx, coord_fin, symbol_fin 

def frag_search(cen_idx,nei_idx,covalent_map,bond,symbol,n_heavy):
    """
    search fragments in the cluster
    """
    visited_atom = []; tot_idx = []
    cluster_idx = [cen_idx] + list(nei_idx)
    covalent_sub = covalent_map[cluster_idx][:,cluster_idx]
    symbol = [symbol[u] for u in cluster_idx]
    def lig(cen_idx, visited_atom, tot_idx, covalent_sub, n_heavy):
        tot_idx.append(cen_idx); visited_atom.append(cen_idx) 
        nei_1 = np.where(covalent_sub[cen_idx] == 1)[0]
        for j in nei_1:
            tot_idx.append(j); visited_atom.append(j)
            nei_1 = np.where(covalent_sub[j] == 1)[0]
            for k in nei_1:
                if k not in visited_atom:
                    tot_idx.append(k); visited_atom.append(k)
                    nei_1 = np.where(covalent_sub[k] == 1)[0]
                else:
                    continue
                for l in nei_1:
                    if l not in visited_atom:
                        tot_idx.append(l); visited_atom.append(l)
                        nei_1 = np.where(covalent_sub[l] == 1)[0]
                    else:
                        continue
                    for m in nei_1:
                        if m not in visited_atom:
                            tot_idx.append(m); visited_atom.append(m)
                            nei_1 = np.where(covalent_sub[m] == 1)[0] 
                        else:
                            continue
                        for u in nei_1:
                            if u not in visited_atom:
                                tot_idx.append(u); visited_atom.append(u)
                                nei_1 = np.where(covalent_sub[u] == 1)[0]
                            else:
                                continue
                            for v in nei_1:
                                if v not in visited_atom:
                                    tot_idx.append(v); visited_atom.append(v)
                                else:
                                    continue

        for i in range(len(covalent_sub)):
            if i not in tot_idx:
                cen_idx = i
        if len(tot_idx) == len(covalent_sub):
            cen_idx = None
        return cen_idx, visited_atom, tot_idx
   
    cen_idx = 0; ligs = []
    while len(tot_idx) < 1 + len(nei_idx):
        cen_idx, visited_atom, tot_idx = lig(cen_idx, visited_atom, tot_idx, covalent_sub, n_heavy)
        ligs.append([cluster_idx[atom] for atom in visited_atom]); visited_atom = []
        if len(tot_idx) == len(nei_idx) + 1:
            assert(cen_idx == None)
    return ligs

if __name__ == '__main__':
    from ase.io import read
    mol = read('polypep.pdb')
    symbol = mol.get_chemical_symbols()
    coord = mol.get_positions()
    bond_n = _crd2frag(symbol, coord)#; bond_n = sorted(bond_n)
    bond_n = (np.array(bond_n)+1)
    for bond in bond_n:
        if bond[0] == 65:
            print(bond)
    
