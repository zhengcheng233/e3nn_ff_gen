#!/usr/bin/env python
"""
submit job in batcth at the legebue platform
"""
import os
from glob import glob 
import numpy as np
import subprocess as sp
import json
import shutil
from typing import List
from collections import Counter
import warnings
import random
import dpdata
from dpgen.remote.decide_machine import convert_mdata
from dpgen.dispatcher.Dispatcher import Dispatcher, _split_tasks, make_dispatcher, make_submission
from dpgen import dlog
import sys

def run_e3nn(mdata,prefix,tensor_name,number_node, train_mode = True):
    convert_mdata(mdata,['train']); all_tasks = []; commands = []
    tensor_name1 = tensor_name.split('.')[0]
    #train_command = "python3 train.py --config config_dipole --config_spec \"{'data_config.n_train':6416,'data_config.n_val':712,'data_config.path':'./%s_%s','batch_size':64,'stride':4}\"" %(str(prefix),str(tensor_name))
    #train_command = "python3 train.py --config config_monopole --config_spec \"{'data_config.n_train':6416,'data_config.n_val':712,'data_config.path':'./%s_%s','batch_size':64,'stride':4}\"" %(str(prefix),str(tensor_name))
    #train_command = "python3 train.py --config config_quadrupole --config_spec \"{'data_config.n_train':6416,'data_config.n_val':712,'data_config.path':'./%s_%s','batch_size':64,'stride':4}\"" %(str(prefix),str(tensor_name))
    train_command = 'molpro -n 4 input.inp'
    #train_command = "python3 inference.py --config config_monopole --config_spec \"{'data_config.path':'./%s_%s','batch_size':64}\" --model_path results_%s/default_project/default/best.pt --output_keys monopole --output_path %s_pred.hdf5" %(str(prefix),str(tensor_name),str(tensor_name1),str(tensor_name1))
    mdata['train_command'] = train_command
    mdata['train_resources']['number_node'] = number_node
    train_group_size = mdata['train_group_size']
    train_resources = mdata['train_resources']
    train_machine = mdata['train_machine']
    make_failure = train_resources.get('mark_failure', False)
    work_path = os.path.join("./")
    task_path = os.path.join(work_path,prefix)
    all_tasks.append(task_path)
    commands.append(train_command)
    forward_files = ['input.inp']
    #if train_mode == True:
    #    forward_files = ['train.py']
    #else:
    #    forward_files = ['inference.py']
    #    forward_files += ['results_'+str(tensor_name1)]
    backward_files = ['input.out']

    submission = make_submission(
          train_machine,
          train_resources,
          commands=commands,
          work_path=work_path,
          run_tasks=all_tasks,
          group_size=train_group_size,
          forward_common_files=[],
          forward_files=forward_files,
          backward_files=backward_files,
          outlog = 'train.log',
          errlog = 'train.log')
    submission.run_submission()

mdata = json.load(open('machine.json'))
prefix = 'test' 
tensor_name = 'none' 
number_node = 1233334
run_e3nn(mdata,prefix,tensor_name,number_node,train_mode=False)
#f_name = int(sys.argv[1])
#number_node = int(sys.argv[2])
#cwd_ = os.getcwd()
#f_dirs = [glob('./*/*c8.hdf5')[f_name]]
#for f_name in f_dirs:
#    file_name = os.path.basename(f_name)
#    f_name = f_name[:-len(file_name)]
    #os.chdir(f_name)
#    prefix = f_name.split('/')[-2]
#    mdata = json.load(open('machine.json'))
#    tensor_name = file_name.split('_')[-1]
#    run_e3nn(mdata,prefix,tensor_name,number_node,train_mode=False)
    #os.chdir(cwd_)
