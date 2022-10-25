#!/usr/bin/env python
import ase

def dumpposcar(pos,sym,f_name,type_map):
    atomic_n = list(ase.atom.atomic_numbers.keys())
    type_idx = {}
    for idx,ele in enumerate(type_map):
        type_idx[ele] = idx + 1
    with open('conf.dump','w') as fp:
        fp.write('ITEM: ATOMS id type x y z fx fy fz'+'\n')
        for ii in range(len(sym)):
            fp.write('%s %s %s %s %s' %(str(ii+1),str(type_idx[atomic_n[sym[ii]]]),str(pos[ii][0]),str(pos[ii][1]),str(pos[ii][2]))+'\n')

    atomic_n = list(ase.atom.atomic_numbers.keys())
    sym = [atomic_n[s] for s in sym]
    coord = []; symbol = []
    ele_num = {}
    for ele in type_map:
        ele_num[ele] = 0

    for ele in type_map:
        for cc,ss in zip(pos,sym):
            if ss == ele:
                coord.append(cc); symbol.append(ss)
                ele_num[ele] += 1
    with open(f_name,'w') as fp:
        for ele in type_map:
            if ele_num[ele] > 0:
                fp.write(ele+'\t')
        fp.write('\n')
        for ele in type_map:
            if ele_num[ele] > 0:
                fp.write(str(ele_num[ele])+'\t')
        fp.write('\n')
        for cc in coord:
            fp.write('  %.10f   %.10f   %.10f' %(cc[0], cc[1], cc[2])+'\n')

def reloadposcar(f_name):
    sys_data = {}; coord = []; atom_names = []; atom_types = []
    with open(f_name,'r') as fp:
        for idx,line in enumerate(fp):
            line = line.strip().split()
            if idx == 0:
                atom_names = [u for u in line]
            elif idx == 1:
                for ii,uu in enumerate(line):
                    atom_types.extend([ii]*int(uu))
            else:
                coord.append([float(line[0]),float(line[1]),float(line[2])])
    sys_data['coords'] = [coord]
    sys_data['atom_names'] = atom_names
    sys_data['atom_types'] = atom_types
    return sys_data
