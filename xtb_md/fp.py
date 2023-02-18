#!/usr/bin/env python
"""
make a relative general script for xtb submit
"""

import os
import json
from glob import glob 

def make_fp(files,mdata,task_name):
    # generate slurm scripts for fp calculation, including s0, s1 opt, soc calculation, momap calculation 
    # generate python script during the calc 
    # generate slurm script and input file
    nproc = mdata['fp']['nproc']
    task_num = mdata['fp']['task_num']
    machine_queue = mdata['fp']['queue']
    user_name = mdata['fp']['user_name']
    task_set = [[] for i in range(task_num)]
    _cwd = os.getcwd()
    for ii in range(0, len(files)):
        task_set[ii % task_num].append(files[ii])
    for idx, ii in enumerate(task_set):
        with open(task_name+'_'+str(idx)+'.slurm','w') as fp:
            fp.write('#!/bin/bash'+'\n')
            fp.write('#SBATCH -p '+str(machine_queue)+'\n')
            fp.write('#SBATCH -J ' + task_name + '_'+str(idx)+'\n')
            fp.write('#SBATCH -N 1'+'\n')
            fp.write('#SBATCH -n '+str(nproc)+'\n')
            #fp.write('#SBATCH --exclusive'+'\n')
            fp.write('mkdir -p /tmp/scratch/'+str(user_name)+'/'+task_name+'.$SLURM_JOB_ID'+'\n')
            for jj in ii:
                file_name = os.path.basename(jj)
                abs_path = os.path.abspath(jj)[:-len(file_name)]
                tmp_path = '/tmp/scratch/'+str(user_name)+'/'+task_name+'.$SLURM_JOB_ID/'+jj[2:-len(file_name)]
                dir_path = tmp_path 
                fp.write('mkdir -p ' + dir_path + '\n')
                fp.write('cp '+abs_path+'dimer_sapt.gjf '+ tmp_path+'\n')
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/xtb_md.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                 
                file_name = os.path.basename(jj)
                fp.write('cd /tmp/scratch/'+str(user_name)+'/'+task_name+'.$SLURM_JOB_ID'+'\n')
                fp.write('cd '+str(jj[:-len(file_name)])+'\n')
                fp.write('python xtb_md.py > tmp.log'+'\n')
                file_name = os.path.basename(jj) 
                abs_path = os.path.abspath(jj)[:-len(file_name)]
                tmp_path = '/tmp/scratch/'+str(user_name)+'/'+task_name+'.$SLURM_JOB_ID/'+jj[2:-len(file_name)]
                fp.write('cp '+tmp_path+'coord.npy ' + abs_path+'\n')
                fp.write('sleep 0.5'+'\n')
                fp.write('rm -rf '+tmp_path+'\n')
            fp.write('rm -rf '+'/tmp/scratch/'+str(user_name)+'/'+task_name+'.$SLURM_JOB_ID'+'\n')
    return task_set 
        
def calc_fp(task_set):
    for idx in range(len(task_set)):
        os.system('sbatch momap_'+str(idx)+'.slurm')
    return 

def post_fp(files):
    pass
    return 

if __name__ == '__main__':
    #files = glob('./iter.init/02.fp/conf.*/input.com')[710:]
    files = glob('./conf.*/dimer_sapt.gjf')[0:100] 
    with open('machine.json','r') as fp:
        mdata = json.load(fp)
    task_set = make_fp(files,mdata,'xtb') 
    #calc_fp(task_set)
