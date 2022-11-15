#!/usr/bin/env python
"""
generate dimer using scan, dimer may include monomers with different types. fps+soap with be used to sample 
initial config later, model_devation with xtb traj will used to cerity which initial config should be used 
for later e3nn md explore 
"""

import sys
import numpy as np
import MDAnalysis as mda
import copy
import os

def get_dimer(f_name):
    data = np.load(f_name,allow_pickle=True)
    coords = data['coords']; symbols = data['symbols']
    monA_idx = data['monA_idx']; monB_idx = data['monB_idx']
    bondA_idx = data['bondA_idx']; bondB_idx = data['bondB_idx']
    ip_a = data['ip_a']; ip_b = data['ip_b']
    homo_a = data['homo_a']; homo_b = data['homo_b']
    return coords, symbols, monA_idx, monB_idx, bondA_idx, bondB_idx, ip_a, ip_b, homo_a, homo_b

def gen_molpro_output(coord, symbol, monA_idx, monB_idx, template_fn, ac_data, ofn=None, label=None):
    if ofn is None:
        ofile = sys.stdout
    else:
        ofile = open(ofn, 'w')
    if label is not None:
        print('! %s'%label, file=ofile)
    indices_A = np.array(range(1, len(monA_idx)+1))
    indices_B = np.array(range(len(monA_idx)+1, len(monA_idx)+len(monB_idx)+1))
    n_atoms_tot = len(monA_idx) + len(monB_idx)
    r_midbond = mass_center(coord,symbol)
    with open(template_fn) as ifile:
        iread = 0
        for line in ifile:
            if iread == 0 and 'geometry=' in line:
                iread = 1
                print(line, end='', file=ofile)
                continue
            if iread == 1 and '}' in line:
                i_atom = 1
                for ele,r in zip(symbol,coord):
                    print('%d,%s,,%.8f,%.8f,%.8f'%(i_atom, ele, r[0], r[1], r[2]), file=ofile)
                    i_atom += 1
                print('%d,He,,%.8f,%.8f,%.8f'%(i_atom, r_midbond[0], r_midbond[1], r_midbond[2]), file=ofile)
                print(line, end='', file=ofile)
                iread = 0
                continue
            elif iread == 1:
                continue
            if iread == 0 and '!monomer A' in line:
                print(line, end='', file=ofile)
                iread = 2
                continue
            elif iread == 0 and '!monomer B' in line:
                print(line, end='', file=ofile)
                iread = 3
                continue
            elif iread == 0 and '!dimer' in line:
                print(line, end='', file=ofile)
                iread = 4
                continue
            if iread == 2:
                print('dummy', end='', file=ofile)
                for i in indices_B:
                    print(',%d'%i, end='', file=ofile)
                print(',%d'%(n_atoms_tot+1), file=ofile)
                print('', file=ofile)
                iread = 0
                continue
            elif iread == 3:
                print('dummy', end='', file=ofile)
                for i in indices_A:
                    print(',%d'%i, end='', file=ofile)
                print(',%d'%(n_atoms_tot+1), file=ofile)
                print('', file=ofile)
                iread = 0
                continue
            elif iread == 4:
                print('dummy,%d'%(n_atoms_tot+1), file=ofile)
                iread = 0
                continue
            if 'ip_A=' in line:
                print('ip_A=%.6f'%ac_data[0, 0], file=ofile)
                continue
            elif 'eps_homo_pbe0_A=' in line:
                print('eps_homo_PBE0_A=%.6f'%ac_data[0, 1], file=ofile)
                continue
            elif 'ip_B=' in line:
                print('ip_B=%.6f'%ac_data[1, 0], file=ofile)
                continue
            elif 'eps_homo_pbe0_B=' in line:
                print('eps_homo_PBE0_B=%.6f'%ac_data[1, 1], file=ofile)
                continue
            print(line, end='', file=ofile)
    ofile.close()
    return

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

def mass_center(coord,symbol):
    symbol = symbol.reshape(-1,1)
    return np.sum(coord * symbol, axis=0)/np.sum(symbol)
   

def gen_scan(coord, symbol, monA_idx, monB_idx, bondA_idx, bondB_idx):
    coord_A = [coord[uu] for uu in monA_idx]; coord_B = [coord[uu] for uu in monB_idx]
    symbol_A = [symbol[uu] for uu in monA_idx]; symbol_B = [symbol[uu] for uu in monB_idx]
    dr = 1; r_min = 1.4; r_max = 6.2
    n_atoms1 = len(monA_idx); n_atoms2 = len(monB_idx)
    i, j, min_dr = find_closest_distance(coord_A, coord_B) 
    dr_com = mass_center(coord_B, symbol_A) - mass_center(coord_A,symbol_A)
    dn_com = dr_com / np.linalg.norm(dr_com)
    pos0 = copy.deepcopy(coord_B)
    i = 0; di = 0.1
    while min_dr > r_min:
        i -= di
        pos = copy.deepcopy(pos0)
        pos += dn_com * dr * i
        _, _, min_dr = find_closest_distance(coord_A, pos)
        if min_dr < r_min:
            break
    i_min = i + di
    i = 0 
    while min_dr < r_max:
        i += di 
        pos = copy.deepcopy(pos0)
        pos += dn_com * dr * i
        _, _, min_dr = find_closest_distance(coord_A, pos)
        if min_dr > r_max:
            break
    i_max = i - di
    i_switch1 = i_min + (i_max - i_min)/6 
    i_switch2 = i_min + (i_max - i_min)*3/6
    indices = list(np.arange(i_min, i_swithc1, (i_switch1-i_min)/4)) \
            + list(np.arange(i_switch1, i_switch2, (i_swith2 - i_switch1)/4)) \
            + list(np.arange(i_switch2, i_max, (i_max-i_switch2)/3))
    positons = []
    for i in indices:
        pos = copy.deepcopy(pos0)
        pos += dn_com * dr * i
        coord[-len(pos):] = pos
        positions.append(coord)
    return indices, positions
    
def gen_gjf(pos, symbol,ofn=None):
    if ofn is None:
        ofile = sys.stdout
    else:
        ofile = open(ofn, 'w')
    print("# HF/6-31G(d)\n\ntitle\n\n0 1", file=ofile)
    for elem,r in zip(symbol,pos):
        elem = atom.type
        r = atom.position
        print('%3s%15.8f%15.8f%15.8f'%(elem, r[0], r[1], r[2]), file=ofile)
    print('', file=ofile)
    return

def padding(i):
    s = '%d'%i
    while len(s) < 3:
        s = '0' + s
    return s

def clean_folder(i_frame, maindir):
    folder = maindir + '/' + padding(i_frame)
    if os.path.isdir(folder):
        os.system('rm -r %s'%folder)
    os.system('mkdir %s'%folder)
    return folder

if __name__ == '__main__':
    # pick a frame and scan
    # i_frame = np.random.randint(1000) #int(sys.argv[1])
    f_name = sys.argv[1]; i_frame = int(sys.argv[2])
    coords, symbols, monA_idx, monB_idx, bondA_idx, bondB_idx, ip_a, ip_b, homo_a, homo_b = get_dimer(f_name)
    # scan the dimer geometry provided according to i_frame
    coord = coords[i_frame]; symbol = symbols[i_frame]
    indices, positions = gen_scan(coord,symbol,monA_idx,monB_idx,bondA_idx,bondB_idx) 

    n_data = len(indices)
    folder_gjf = clean_folder(i_frame, 'gjfs')
    folder_sapt = clean_folder(i_frame, 'sapt')
    folder_mp2 = clean_folder(i_frame, 'mp2')
    folder_pdb = clean_folder(i_frame, 'pdb')
    for i_data in range(n_data):
        pos = positions[i_data]
         

        # generate gjf files for visualization
        ofn = folder_gjf + '/' + padding(i_data) + '.gjf'
        gen_gjf(pos, symbol, ofn)

        # write pdb
        #ofn = folder_pdb + '/' + padding(i_data) + '.pdb'
        #u.atoms.write(ofn)

        # generate sapt file
        ofn = folder_sapt + '/' + padding(i_data) + '.com'
        gen_molpro_output(pos, symbol, monA_idx, monB_idx, 'sapt_template.com', np.array([[ip_a,homo_a],[ip_b,homo_b]]), ofn=ofn, label='shift= %.6f'%indices[i_data])

        # # generate mp2 file
        ofn = folder_mp2 + '/' + padding(i_data) + '.com'
        gen_molpro_output(pos, symbol, monA_idx, monB_idx, 'mp2_template.com', np.array([[ip_a,homo_a],[ip_b,homo_b]]), ofn=ofn, label='shift = %.6f'%indices[i_data])
