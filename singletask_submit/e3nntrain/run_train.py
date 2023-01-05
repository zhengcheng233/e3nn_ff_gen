#!/usr/bin/env python

import h5py
import os
import sys
import argparse
import glob
import json
import random
import logging
import logging.handlers
import queue
import warnings
import shutil
import time
import copy
import dpdata
import numpy as np
import subprocess as sp
import scipy.constants as pc
from collections import Counter
from distutils.version import LooseVersion
from typing import List
from numpy.linalg  import norm
from dpgen import dlog
from dpgen import SHORT_CMD
from dpgen.generator.lib.utils import make_iter_name
from dpgen.generator.lib.utils import create_path
from dpgen.generator.lib.utils import copy_file_list
from dpgen.generator.lib.utils import replace
from dpgen.generator.lib.utils import log_iter
from dpgen.generator.lib.utils import record_iter
from dpgen.generator.lib.utils import log_task
from dpgen.generator.lib.utils import symlink_user_forward_files
from dpgen.generator.lib.lammps import make_lammps_input, get_dumped_forces
from dpgen.generator.lib.gaussian import make_gaussian_input, take_cluster
from dpgen.remote.decide_machine import convert_mdata
from dpgen.dispatcher.Dispatcher import Dispatcher, _split_tasks, make_dispatcher, make_submission
from dpgen.util import sepline
from dpgen import ROOT_PATH
from dpgen.generator import run
import random

template_name = 'template'
train_name = '00.train'
train_task_fmt = '%03d'
train_tmpl_path = os.path.join(template_name, train_name)
default_train_input_file = 'input.json'
data_system_fmt = '%03d'
model_devi_name = '01.model_devi'
model_devi_task_fmt = data_system_fmt + '.%06d'
model_devi_conf_fmt = data_system_fmt + '.%04d'
fp_name = '02.fp'
fp_task_fmt = data_system_fmt + '.%06d'

_make_model_devi_revmat = run._make_model_devi_revmat; _make_model_devi_native_gromacs = run._make_model_devi_native_gromacs
_link_fp_vasp_pp = run._link_fp_vasp_pp; post_fp_check_fail = run.post_fp_check_fail
make_model_devi_task_name = run.make_model_devi_task_name; make_model_devi_conf_name = run.make_model_devi_conf_name
make_fp_task_name = run.make_fp_task_name; poscar_natoms = run.poscar_natoms
poscar_shuffle = run.poscar_shuffle; expand_idx = run.expand_idx
parse_cur_job = run.parse_cur_job; dump_to_poscar = run.dump_to_poscar 
dump_to_deepmd_raw = run.dump_to_deepmd_raw; run_fp_inner = run.run_fp_inner
_gaussian_check_fin = run._gaussian_check_fin; _check_skip_train = run._check_skip_train
_check_empty_iter = run._check_empty_iter; detect_batch_size = run.detect_batch_size

ii = int(sys.argv[1]); epoch_sub = int(sys.argv[2])
work_path = './'
task_path = os.path.join(train_task_fmt % ii)
all_tasks = []
all_tasks.append(task_path)
n_train = 0 
from monty.serialization import loadfn
jdata = loadfn('param.json')
mdata = loadfn('machine.json')
mdata = convert_mdata(mdata)
mdata['train'][0]['resources']['number_node'] = random.randint(0,10000000) 
 
#init_data_sys_ = jdata['init_data_sys']
init_data_sys = []
#for ii in init_data_sys_ :
#    init_data_sys.append(os.path.join('data.all/data.init', ii))
trains_comm_data = []
cwd = os.getcwd()
os.chdir(work_path)
fp_data = glob.glob(os.path.join('data.all/data*','iter.*','02.fp','data.*'))
#for ii in init_data_sys :
#    trains_comm_data += glob.glob(os.path.join(ii, 'fp.hdf5'))
#    n_train += h5py.File(os.path.join(ii, 'fp.hdf5'),'r')['_n_nodes'].shape[0]
for ii in fp_data:
    if os.path.isfile(os.path.join(ii, 'fp.hdf5')):
        trains_comm_data += glob.glob(os.path.join(ii, 'fp.hdf5'))
        n_train += h5py.File(os.path.join(ii, 'fp.hdf5'),'r')['_n_nodes'].shape[0]
os.chdir(cwd)
#n_train = int(n_train)

#n_train = int(sys.argv[3])

training_init_model = False; train_command = "config_energy_force"
commands = []
if training_init_model:
    command = "python3 train.py --config %s --config_spec \"{'data_config.n_train':%s,'data_config.n_val':64,'data_config.path':'../data.all/:.+fp.hdf5','batch_size':64,'stride':%s, 'epoch_subdivision':%s}\" --resume_from results/default_project/default/trainer.pt" %(train_command, n_train-64, max(int(n_train/10000),1),epoch_sub)
else:
    command = "python3 train.py --config %s --config_spec \"{'data_config.n_train':%s,'data_config.n_val':64,'data_config.path':'../data.all/:.+fp.hdf5','batch_size':64,'stride':%s, 'epoch_subdivision':%s}\"" %(train_command, n_train-64, max(int(n_train/10000),1),epoch_sub)
commands.append(command)

run_tasks = [os.path.basename(ii) for ii in all_tasks]
forward_files = ['train.py']
if training_init_model:
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'trainer.pt')]
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'last.pt')]
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'best.pt')]
backward_files = ['results']

train_group_size = 1

user_forward_files = mdata.get("train" + "_user_forward_files", [])
forward_files += [os.path.basename(file) for file in user_forward_files]
backward_files += mdata.get("train" + "_user_backward_files", [])

submission = make_submission(
     mdata['train_machine'],
     mdata['train_resources'],
     commands=commands,
     work_path=work_path,
     run_tasks=run_tasks,
     group_size=train_group_size,
     forward_common_files=trains_comm_data,
     forward_files=forward_files,
     backward_files=backward_files,
     outlog = 'train.log',
     errlog = 'train.log')

submission.run_submission()

