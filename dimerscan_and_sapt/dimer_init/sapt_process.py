#!/usr/bin/env python
"""
this script is used to post-process some sapt tasks, for example link the homo and ionization energy 
"""
import os
import json
from glob import glob 

def sym_link(f_name):
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
    return 

def make_sapt(conf_name, task_name):
    # generate sapt tasks 
    fmt0 = '%03d'; fmt1 = '%06d'

    f_dirs = glob('./dimer_md_for_sapt/task.*/sapt/*/'+conf_name)
    for dir0 in f_dirs:
        dir_n = dir0.strip().split('/')
        idx_0 = int(dir_n[2].split('.')[1]); idx_1 = int(dir_n[-2])
        f_name = task_name + '.' + fmt0%(idx_0) + '.' + fmt1%(idx_1)
        os.system('mkdir -p ./dimer_md_for_sapt/'+f_name)
        os.system('cp ' + dir0 + ' ./dimer_md_for_sapt/'+f_name+'/input.inp')

if __name__ == '__main__':
    make_sapt('008.com','near')
    # make_sapt('010.com','far')
