#!/usr/bin/env python
import os
import sys
import numpy as np
langevin_temperature = int(sys.argv[1]); steps = float(sys.argv[2])
output_period = int(sys.argv[3])

fr = open('q_net.txt','r'); lines = fr.readlines(); fr.close()
mol_charge = int(lines[0].strip().split()[0])

def make_lammps(ii,n_step,coord,box_latt,atype):
    ptr_float_fmt = '%15.10f'; ptr_int_fmt = '%6d'; ptr_key_fmt = '%15s'

    buff = []
    buff.append('ITEM: TIMESTEP')
    buff.append('%s' %(str(ii * n_step)))
    buff.append('ITEM: NUMBER OF ATOMS')
    buff.append('%s' %(str(len(coord))))
    buff.append('ITEM: BOX BOUNDS xy xz yz ff ff ff')
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append('ITEM: ATOMS id type x y z fx fy fz')
    coord_fmt = ptr_int_fmt + ' ' + ptr_int_fmt + ' '+ ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt
    for ii, r0 in enumerate(coord):
        buff.append(coord_fmt % (ii+1, atype[ii], r0[0], r0[1], r0[2]))
    return '\n'.join(buff)

mass = {'H':1.00794,'C':12.0107,'N':14.0067,'O':15.9994,'S':32.065}
sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
lmp_map = ["C","H","N","O","S"]; flag = False
lmp_type = []; lmp_symbol = []

mol_coords = []; mol_atypes = []; mol_masses = []
with open('conf.lmp','r') as fp:
    for line in fp:
        line = line.strip().split()
        if len(line) == 2 and line[-1] == 'atoms':
            mol_numAtoms = int(line[0])
        elif len(line) == 4 and line[-1] == 'xhi' and line[-2] == 'xlo':
            box_latt = max(float(line[-3]),100.)
        elif len(line) == 3 and line[0] == 'Atoms':
            flag = True
        if flag == True:
            if len(line) == 5:
                mol_coords.append([float(x) for x in line[2:5]])
                mol_atypes.append(sym_dict[lmp_map[int(line[1])-1]])
                mol_masses.append(mass[lmp_map[int(line[1])-1]])
                lmp_type.append(int(line[1])); lmp_symbol.append(lmp_map[int(line[1])-1])

#if os.path.isfile('./traj/file.txt'):
#    pass
#else:
#    os.system('mkdir -p traj')
#    with open('./traj/file.txt','w') as fp:
#        fp.write('created')

with open('conf.xyz','w') as fp:
    fp.write('%s' %(str(len(lmp_symbol)))+'\n')
    fp.write('\n')
    for ii,cc in enumerate(mol_coords):
        fp.write('%s %s %s %s'%(str(lmp_symbol[ii]),str(cc[0]),str(cc[1]),str(cc[2]))+'\n')

with open('md.inp','w') as fp:
    fp.write('$md'+'\n')
    fp.write(' temp=%s' %(str(langevin_temperature))+'\n')
    fp.write(' time=%s' %(str(steps))+'\n')
    fp.write(' dump=%s' %(str(int(output_period)))+'\n')
    fp.write(' step= 0.5'+'\n')
    if langevin_temperature > 500:
        fp.write(' nvt=false'+'\n')
    else:
        fp.write(' nvt=true'+'\n')
    fp.write(' hmass=1'+'\n')
    if langevin_temperature > 500:
        fp.write('shake=2'+'\n')
    else:
        fp.write('shake=0'+'\n')
    fp.write('$end'+'\n')

os.system('xtb conf.xyz --input md.inp --omd --gfn 2 --chrg %s' %(str(mol_charge)))

coords = []; coord = []
with open('xtb.trj','r') as fp:
    for line in fp:
        line0 = line.strip().split()
        if len(line0) == 4:
            coord.append([float(line0[1]),float(line0[2]),float(line0[3])])
        elif 'xtb:' in line and len(coord) > 0:
            coords.append(coord); coord = []
    coords.append(coord)

coords = np.array(coords)
for idx,coord in enumerate(coords):
    coord_cen = np.mean(coord,axis=0)
    coord = coord - coord_cen + np.array([box_latt/2., box_latt/2., box_latt/2.])
    ret = make_lammps(idx,output_period,coord,box_latt,lmp_type)
    with open('./traj/'+str(idx*output_period)+'.lammpstrj','w') as fp:
        fp.write(ret)

