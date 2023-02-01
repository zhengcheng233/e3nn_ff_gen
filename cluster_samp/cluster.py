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
    for i in range(1,7):
        nei_idx = np.where(cov_idx == i)[0]
        for j in nei_idx:
            if n_atom >= n_heavy:
                continue
            else:
                if j in frg_idx:
                    if symbol[j] != 'H':
                        n_atom += 1
                    frg_idx_fin.append(j)

    # add atoms in frg_idx_fin if  
    for ii in frg_idx_fin:
        nei = np.where(cov_idx[ii] == 1)[0]
        for jj in nei:
            if jj not in frg_idx_fin:
                if bond_type[(ii,jj)] > 1:
                    frg_idx_fin.append(jj)
                    nei = np.where(cov_idx[jj] == 1)[0]
                    for kk in nei:
                        if kk not in frg_idx_fin:
                            hydro.append([jj,kk])
                else:
                    hydro_idx.append([ii,jj])

    coord_fin = [coord[u] for u in frg_idx_fin]
    symbol_fin = [symbol[u] for u in frg_idx_fin]
    for ii in hydro_idx:
        symbol_fin.append('H')
        h_coord = add_hydron(coord[ii[0]],coord[ii[1]])
        coord_fin.append(h_coord)
    return frg_idx_fin, hydro_idx, coord_fin, symbol_fin 

def frag_search(cen_idx,nei_idx,covalent_map,bond,symbol):
    """
    search fragments in the cluster
    """
    visited_atom = []; tot_idx = []
    cluster_idx = [cen_idx] + list(nei_idx)
    covalent_sub = covalent_map[cluster_idx][:,cluster_idx]
    symbol = [symbol[u] for u in cluster_idx]
    
    def lig(cen_idx, visited_atom, tot_idx, covalent_sub):
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
        cen_idx, visited_atom, tot_idx = lig(cen_idx, visited_atom, tot_idx, covalent_sub,n_heavy)
        ligs.append([cluster_idx[atom] for atom in visited_atom]); visited_atom = []
        if len(tot_idx) == len(nei_idx) + 1:
            assert(cen_idx == None)
    return ligs

if __name__ == '__main__':
    from ase.io import read
    mol = read('tripeptide.pdb')
    symbol = mol.get_chemical_symbols()
    coord = mol.get_positions()
    solute_n = 29
    coord_n = coord[0:solute_n]; symbol_n = symbol[0:solute_n]
    # using openbabel to get the bond 
    bond_n = _crd2frag(symbol_n, coord_n)
    # check bond by mdanalysis and ourself, for charged system, 
    # both openbabel and mdanalysis need specific the charged position manually
    solute = MDAnalysis.Universe('tripeptide_gas.prmtop','tripeptide_gas.pdb')
    bonds = solute.bonds.indices; symbols = solute.atoms.elements
    bond_n_mda = _bonder(bonds,symbols)
    if len(bond_n) != len(bond_n_mda):
        print('attention, the openbabel miss some bonds')
    # obtain the bond and bond order in sol 
    coord_sol = coord[solute_n:]; symbol_sol = symbol[solute_n:]
    bond_sol = _crd2frag(symbol_sol,coord_sol)
    bond_sol = np.array(bond_sol); bond_n_mda = np.array(bond_n_mda)
    
    # merge the bond data
    bond_sol[:,:-1] += np.max(bond_n)
    bond = list(bond_n_mda) + list(bond_sol)
    # covalent_map = generate_covalent_map(bond_n_mda)
    covalent_map = generate_covalent_map(bond)
    
    # the bond and the covalent map can be used for frag gen
    # here use atom i as example 
    atom_idx = 15
    # step 1, get the distance atoms
    coord_mat = cdist(coord,coord,'euclidean')
    
    # step 2, accoring the model devi order, define the visited atoms, if interval atom i and j  <= 3, added to visited atoms
    # not generate frag

    # step 3, generate frag; may need functional later 
    dis_mat = coord_mat[atom_idx]; nei_idx = np.where(dis_mat < Rc)[0]
    # get the frag with full atom 
    ligs = frag_search(atom_idx,nei_idx,covalent_map,bond,symbol)
    frg_idx_fins = []; hydro_idxs = []; coord_fins = []; symbol_fins = []
    for lig in ligs:
        frg_idx_fin, hydro_idx, coord_fin, symbol_fin = frag_gen(lig[0], lig, covalent_map, coord, symbol, bond, n_heavy = 8)
        frg_idx_fins.append(frg_idx_fin)
        hydro_idxs.append(hydro_idx)
        coord_fins.append(coord_fin)
        symbol_fins.append(symbol_fin)



    
