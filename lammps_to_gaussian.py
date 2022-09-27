#!/usr/bin/env python
import os
from glob import glob
import sys

dir_name = os.path.join('task.'+'%03d'%(int(sys.argv[1]))+'.000000','traj')

dirs = glob(dir_name+'/*.lammpstrj')

ele_type = ['X','C','H','N','O','S']

def gencom(dir0):
    f_name = os.path.basename(dir0).split('.')[0]
    eles = []; coord = []
    with open(f_name+'.lammpstrj') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 5:
                eles.append(int(line[1]))
                coord.append([float(line[2]),float(line[3]),float(line[4])])
    symbol = []
    for ii in eles:
        symbol.append(ele_type[ii])
    with open(f_name+'.com','w') as fp:
        fp.write('%chk=1.chk'+'\n')
        fp.write('# pm6'+'\n')
        fp.write('\n')
        fp.write('dpgen'+'\n')
        fp.write('\n')
        fp.write('1 1'+'\n')
        for ii,cc in enumerate(coord):
            fp.write('%s %s %s %s' %(str(symbol[ii]),str(cc[0]),str(cc[1]),str(cc[2]))+'\n')
        fp.write('\n')

_cwd = os.getcwd()
for dir0 in dirs:
    f_name = os.path.basename(dir0)
    f_dir = dir0[:-len(f_name)]
    os.chdir(f_dir)
    gencom(dir0)
    os.chdir(_cwd)
