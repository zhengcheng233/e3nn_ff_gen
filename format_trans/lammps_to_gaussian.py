#!/usr/bin/env python
import os
from glob import glob

dirs = glob('./*.lammpstrj')

ele_type = ['X','C','H','N','O','S']

def gencom(dir0):
    f_name = os.path.basename(dir0).split('.')[0]
    eles = []; coords = []
    with open(f_name+'.lammpstrj'):
        for line in fp:
            line = line.strip().split()
            if len(line) == 5:
                eles.append(int(line[1]))
                coords.append([float(line[2]),float(line[3]),float(line[4])])
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

for dir0 in dirs:
    gencom(dir0)


