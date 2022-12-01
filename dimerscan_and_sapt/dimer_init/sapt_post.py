#!/usr/bin/env python
"""
this script is used to post-process some sapt tasks, for example link the homo and ionization energy 
"""
import os
import json
from glob import glob 

fmt = '%03d'
with open('param.json','r') as fp:
    param = json.load(fp)

configs_prefix = param['sys_configs_prefix']
sys_configs = param['sys_configs']

cwd_ = os.getcwd()

for idx, sys in enumerate(sys_configs):
    dir0 = os.path.join(configs_prefix,sys[0])
    f_name = os.path.basename(dir0); dir0 = dir0[:-len(f_name)]
    os.system('rm '+os.path.join('./dimer_md_for_sapt','task.'+fmt%(idx)+'.000000','minimal_data.txt'))
    os.symlink(os.path.abspath(os.path.join(dir0,'minimal_data.txt')),os.path.abspath(os.path.join('./dimer_md_for_sapt','task.'+fmt%(idx)+'.000000','minimal_data.txt')))

