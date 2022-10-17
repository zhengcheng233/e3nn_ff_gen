#!/usr/bin/env python
"""
this script is used to merge different datasets
"""
from glob import glob
import os


for ii in range(0,71):
    dirs = glob('iter.'+'%06d'%(ii)+'/02.fp/data.*')
    os.system('mkdir -p init/data3/iter.'+'%06d'%(ii)+'/02.fp')
    for dir0 in dirs:
        f_0 = os.path.abspath(dir0)
        f_name = os.path.basename(f_0)
        f_1 = os.path.abspath('./init/data3')
        iter_name = (f_0.split('/')[-3])
        os.system('cp -r '+f_0+' '+os.path.relpath(os.path.join(f_1,iter_name,'02.fp',f_name)))
