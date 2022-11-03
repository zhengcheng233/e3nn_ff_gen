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

def run_e3nn(mdata,prefix,tensor_name):
    convert_mdata(mdata,['train']); all_tasks = []; commands = []
    train_command = "python3 train.py --config config_dipole --config_spec \"{'data_config.n_train':6561,'data_config.n_val':730,'data_config.path':'./%s_%s','batch_size':64,'stride':4}\"" %(str(prefix),str(tensor_name))
    mdata['train_command'] = train_command
    train_group_size = mdata['train_group_size']
    train_resources = mdata['train_resources']
    train_machine = mdata['train_machine']
    make_failure = train_resources.get('mark_failure', False)
    work_path = os.path.join("./")
    task_path = os.path.join(work_path,prefix)
    all_tasks.append(task_path)
    commands.append(train_command)
    forward_files = ['train.py']
    forward_files += [os.path.join(prefix,prefix+'_'+tensor_name)]
    backward_files = ['results']
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


#cwd_ = os.getcwd()
f_dirs = glob('./*/*multipole.hdf5')[0:1]
for f_name in f_dirs:
    file_name = os.path.basename(f_name)
    f_name = f_name[:-len(file_name)]
    #os.chdir(f_name)
    prefix = f_name.split('/')[-2]
    mdata = json.load(open('machine.json'))
    tensor_name = file_name.split('_')[-1]
    run_e3nn(mdata,prefix,tensor_name)
    #os.chdir(cwd_)
