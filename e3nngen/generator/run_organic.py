#!/usr/bin/env python
"""
init: data
iter:
        00.train
        01.model_devi
        02.fp
        03.data
1. further improve the active learning process as md explore is too expensive
"""

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
from e3_layers.data import Batch
from lib.io import dumpposcar, reloadposcar
import ase


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
dump_to_deepmd_raw = run.dump_to_deepmd_raw; _gaussian_check_fin = run._gaussian_check_fin
_check_skip_train = run._check_skip_train; _check_empty_iter = run._check_empty_iter
detect_batch_size = run.detect_batch_size


def copy_model(numb_model, prv_iter_index, cur_iter_index, generalML, nequip_md, xtb_md) :
    cwd = os.getcwd()
    prv_train_path = os.path.join(make_iter_name(prv_iter_index), train_name)
    cur_train_path = os.path.join(make_iter_name(cur_iter_index), train_name)
    prv_train_path = os.path.abspath(prv_train_path)
    cur_train_path = os.path.abspath(cur_train_path)
    create_path(cur_train_path)
    for ii in range(numb_model):
        prv_train_task = os.path.join(prv_train_path, train_task_fmt%ii)
        os.chdir(cur_train_path)
        os.symlink(os.path.relpath(prv_train_task), train_task_fmt%ii)
        os.symlink(os.path.join(train_task_fmt%ii, 'results', 'default_project', 'default', 'best.pt'),'best.%03d.pt' % ii) 
        os.chdir(cwd)
        if generalML == True and ii == 0:
            if nequip_md == True:
                task_general_file = os.path.join('./generalpb','general_graph.00'+str(ii)+'.pt')
                task_general_file = os.path.abspath(task_general_file)
                general_ofile = os.path.join(cur_train_path, 'general_graph.%03d.pt' %ii)
                os.symlink(task_general_file, general_ofile)
            elif xtb_md == True:
                general_ofile = []
            else:
                task_general_file = os.path.join('./generalpb','general_graph.00'+str(ii)+'.pb')
                task_general_file = os.path.abspath(task_general_file)
                general_ofile = os.path.join(cur_train_path, 'general_graph.%03d.pb' % ii)
                os.symlink(task_general_file, general_ofile)
    with open(os.path.join(cur_train_path, "copied"), 'w') as fp:
        None

        
def _link_old_models(work_path, old_model_files, ii):
    """
    link the `ii`th old model given by `old_model_files` to
    the `ii`th training task in `work_path`
    """
    task_path = os.path.join(work_path, train_task_fmt % ii)
    task_old_path = os.path.join(task_path, 'old')
    create_path(task_old_path)
    cwd = os.getcwd()
    for jj in old_model_files:
        absjj = os.path.abspath(jj)
        basejj = os.path.basename(jj)
        os.chdir(task_old_path)
        os.symlink(absjj,basejj)
        #os.symlink(os.path.relpath(absjj), basejj)
        os.chdir(cwd)


def write_model_devi_out(devi: np.ndarray, fname: str):
    assert devi.shape[1] == 7
    header = "%10s" % "step"
    for item in 'vf':
        header += "%19s%19s%19s" % (f"max_devi_{item}", f"min_devi_{item}", f"avg_devi_{item}")
    np.savetxt(fname,
               devi,
               fmt=['%12d'] + ['%19.6e' for _ in range(6)],
               delimiter='',
               header=header)
    return devi

def calc_model_devi_f(fs: np.ndarray):
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f

def calc_model_devi(f0,f1,f2,f3,f_name,frequency,reasons):
    forces = [f0,f1,f2,f3]; forces = np.array(forces)
    devi = [np.arange(f0.shape[0]) * frequency]
    devi0 = calc_model_devi_f(forces)
    devi += devi0; devi += devi0; devi = np.vstack(devi).T
    # set unreasonable structures model_devi as very large value
    unreason = False
    reasons = np.array(reasons)
    unreason_ratio = len(np.where(reasons[:,0] < 0.5)[0])/len(reasons)
    for idx,reason in enumerate(reasons):
        if reason[0] == 0 and reason[1] == 0:
            if unreason_ratio < 0.05 or (unreason_ratio < 0.1 and reasons[-1][0] > 0.5):
                devi[idx][1:] = 1000.
            else:
                unreason = True
        elif reason[0] == 0 and reason[1] == 1:
            devi[idx][1:] = 1000.
        if unreason == True:
            devi[idx][1:] = 1000.
    write_model_devi_out(devi,f_name)
    return 

def process_model_devi(all_tasks,freq,f_name):
    cwd = os.getcwd()
    for idx, task in enumerate(all_tasks):
        os.chdir(task)
        f0 = h5py.File('f_pred0.hdf5','r')['forces'][:]; f1 = h5py.File('f_pred1.hdf5','r')['forces'][:]
        f2 = h5py.File('f_pred2.hdf5','r')['forces'][:]; f3 = h5py.File('f_pred3.hdf5','r')['forces'][:]
        n_frame = h5py.File('traj.hdf5','r')['energy'].shape[0]; n_atoms = f0.shape[0]
        f0 = f0.reshape((n_frame,int(n_atoms/n_frame),3)); f1 = f1.reshape((n_frame,int(n_atoms/n_frame),3))
        f2 = f2.reshape((n_frame,int(n_atoms/n_frame),3)); f3 = f3.reshape((n_frame,int(n_atoms/n_frame),3))
        reasons = []
        with open('reasonable.txt','r') as fp:
            for line in fp:
                line = line.strip().split()
                reasons.append([int(line[0]),int(line[1])])
        calc_model_devi(f0,f1,f2,f3,f_name,freq,reasons)
        os.chdir(cwd)

def make_train(iter_index,jdata,mdata):
    train_input_file = default_train_input_file
    numb_models = jdata['numb_models']
    init_data_prefix = jdata['init_data_prefix']
    init_data_prefix = os.path.abspath(init_data_prefix)
    init_data_sys_ = jdata['init_data_sys']
    fp_task_min = jdata['fp_task_min']
    model_devi_jobs = jdata['model_devi_jobs']
    training_iter0_model = jdata.get('training_iter0_model_path', [])
    training_init_model = jdata.get('training_init_model',False)
    training_reuse_iter = jdata.get('training_reuse_iter')
    training_reuse_old_ratio = jdata.get('training_reuse_old_ratio',None)
    if 'training_reuse_stop_batch' in jdata.keys():
        training_reuse_stop_batch = jdata['training_reuse_stop_batch']
    elif 'training_reuse_numb_steps' in jdata.keys():
        training_reuse_stop_batch = jdata['training_reuse_numb_steps']
    else:
        training_reuse_stop_batch = 400000

    nequip_md = jdata.get('nequip_md',True)
    xtb_md = jdata.get('xtb_md',True)
    training_reuse_start_lr = jdata.get('training_reuse_start_lr', 1e-4)
    training_reuse_start_pref_e = jdata.get('training_reuse_start_pref_e', 0.1)
    training_reuse_start_pref_f = jdata.get('training_reuse_start_pref_f', 100)
    model_devi_activation_func = jdata.get('model_devi_activation_func', None)

    generalML = jdata.get('generalML',True)
    if iter_index > 0 and _check_empty_iter(iter_index-1, fp_task_min) :
        log_task('prev data is empty, copy prev model')
        copy_model(numb_models, iter_index-1, iter_index, generalML, nequip_md, xtb_md)
        return
    elif iter_index > 0 and _check_skip_train(model_devi_jobs[iter_index-1]):
        log_task('skip training at step %d ' % (iter_index-1))
        copy_model(numb_models, iter_index-1, iter_index, generalML, nequip_md, xtb_md)
        return
    else :
        iter_name = make_iter_name(iter_index)
        work_path = os.path.join(iter_name, train_name)
        copy_flag = os.path.join(work_path, 'copied')
        if os.path.isfile(copy_flag) :
            os.remove(copy_flag)

    # establish work path
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    create_path(work_path)
    
    # link train.py, init data and iter data; need make change for e3_layer here
    cwd = os.getcwd()
    os.chdir(work_path)
    #os.symlink(os.path.relpath(os.path.join(cwd, 'train.py')), 'train.py')
    os.mkdir('data.all'); os.chdir('data.all')
    os.symlink(os.path.abspath(init_data_prefix), 'data.init')
    os.mkdir('data.iters'); os.chdir('data.iters')
    for ii in range(iter_index):
        os.symlink(os.path.relpath(os.path.join(cwd, make_iter_name(ii))), make_iter_name(ii))
    os.chdir(cwd)

    # !!! old param generate for dp
    ####################################################################
    #init_data_sys = []
    #init_batch_size = []
    #if 'init_batch_size' in jdata:
    #    init_batch_size_ = list(jdata['init_batch_size'])
    #    if len(init_data_sys_) > len(init_batch_size_):
    #        warnings.warn("The batch sizes are not enough. Assume auto for those not spefified.")
    #        init_batch_size.extend(["auto" for aa in range(len(init_data_sys_)-len(init_batch_size))])
    #else:
    #    init_batch_size_ = ["auto" for aa in range(len(jdata['init_data_sys']))]
    #if 'sys_batch_size' in jdata:
    #    sys_batch_size = jdata['sys_batch_size']
    #else:
    #    sys_batch_size = ["auto" for aa in range(len(jdata['sys_configs']))]

    # make sure all init_data_sys has the batch size -- for the following `zip`
    #assert (len(init_data_sys_) <= len(init_batch_size_))
    #for ii, ss in zip(init_data_sys_, init_batch_size_) :
    #    if jdata.get('init_multi_systems', False):
    #        for single_sys in os.listdir(os.path.join(work_path, 'data.init', ii)):
    #            init_data_sys.append(os.path.join('..', 'data.init', ii, single_sys))
    #            init_batch_size.append(detect_batch_size(ss, os.path.join(work_path, 'data.init', ii, single_sys)))
    #    else:
    #        init_data_sys.append(os.path.join('..', 'data.all', 'data.init', ii))
    #        init_batch_size.append(detect_batch_size(ss, os.path.join(work_path, 'data.all', 'data.init', ii)))
    
    #old_range = None
    # I will try to save fp data as dp format and also save hangrui format 
    #if iter_index > 0 :
    #    for ii in range(iter_index) :
    #        if ii == iter_index - 1:
    #            old_range = len(init_data_sys)
    #        fp_path = os.path.join(make_iter_name(ii), fp_name)
    #        fp_data_sys = glob.glob(os.path.join(fp_path, "data.*"))
    #        for jj in fp_data_sys :
    #            sys_idx = int(jj.split('.')[-1])

    #            nframes = dpdata.System(jj,'deepmd/npy').get_nframes()
    #            if nframes < fp_task_min:
    #                log_task('nframes (%d) in data sys %s is too small, skip' % (nframes, jj))
    #                continue
    #            init_data_sys.append(os.path.join('..','data.all','data.iters',jj))
    #            init_batch_size.append(detect_batch_size(sys_batch_size[sys_idx],jj))

    # here, only supprot trainig prarm for hangrui 
    #jinput = jdata['default_training_param']
    #try:
    #    mdata["deepmd_version"]
    #except KeyError:
    #    mdata = set_version(mdata)
    #if LooseVersion(mdata["deepmd_version"]) >= LooseVersion('1') and LooseVersion(mdata["deepmd_version"]) < LooseVersion('2'):
    #    jinput['training']['systems'] = init_data_sys
    #else:
    #    jinput['training']['training_data'] = {}
    #    jinput['training']['training_data']['systems'] = init_data_sys
    #if training_reuse_iter is not None and iter_index >= training_reuse_iter:
    #    if LooseVersion('1') <= LooseVersion(mdata["deepmd_version"]) < LooseVersion('2'):
    #        jinput['training']['stop_batch'] = training_reuse_stop_batch
    #        jinput['training']['auto_prob_style'] \
    #            ="prob_sys_size; 0:%d:%f; %d:%d:%f" \
    #            %(old_range, training_reuse_old_ratio, old_range, len(init_data_sys), 1.-training_reuse_old_ratio)
    #    elif LooseVersion('2') <= LooseVersion(mdata["deepmd_version"]) < LooseVersion('3'):
    #        jinput['training']['numb_steps'] = training_reuse_stop_batch
    #        jinput['training']['training_data']['auto_prob'] \
    #            ="prob_sys_size; 0:%d:%f; %d:%d:%f" \
    #            %(old_range, training_reuse_old_ratio, old_range, len(init_data_sys), 1.-training_reuse_old_ratio)
    #    else:
    #        raise RuntimeError("Unsupported DeePMD-kit version: %s" % mdata["deepmd_version"])
    #    if jinput['loss'].get('start_pref_e') is not None:
    #        jinput['loss']['start_pref_e'] = training_reuse_start_pref_e
    #    if jinput['loss'].get('start_pref_f') is not None:
    #        jinput['loss']['start_pref_f'] = training_reuse_start_pref_f
    #    jinput['learning_rate']['start_lr'] = training_reuse_start_lr
    ####################################################################################################################

    for ii in range(numb_models) :
        task_path = os.path.join(work_path, train_task_fmt % ii)
        create_path(task_path)
        os.chdir(task_path)
        os.symlink(os.path.abspath(os.path.join(cwd,'train.py')),'train.py')
        #for jj in init_data_sys :
        #    if not os.path.isdir(jj) :
        #        raise RuntimeError ("data sys %s does not exists, cwd is %s" % (jj, os.getcwd()))
        os.chdir(cwd)
        #if LooseVersion(mdata["deepmd_version"]) >= LooseVersion('1') and LooseVersion(mdata["deepmd_version"]) < LooseVersion('3'):
        #    jinput['model']['descriptor']['seed'] = random.randrange(sys.maxsize) % (2**32)
        #else:
        #     raise RuntimeError("DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!" )
        
        # dump the input.json !!!!!!!!!!!! need modify later, to generate parameter for nequip
        #with open(os.path.join(task_path, train_input_file), 'w') as outfile:
        #    json.dump(jinput, outfile, indent = 4)

    # link old models link nequip models (here, I use the best.pt) 
    if iter_index > 0 :
        prev_iter_name = make_iter_name(iter_index-1)
        prev_work_path = os.path.join(prev_iter_name, train_name)
        for ii in range(numb_models) :
            prev_task_path =  os.path.join(prev_work_path, train_task_fmt%ii)
            old_model_files = glob.glob(
                os.path.join(prev_task_path, "results"))
            _link_old_models(work_path, old_model_files, ii)
    else:
        prev_iter_name = make_iter_name(iter_index)
        prev_work_path = os.path.join(prev_iter_name, train_name + '.bk000')
        for ii in range(numb_models):
            prev_task_path = os.path.join(prev_work_path, train_task_fmt%ii)
            old_model_files = glob.glob(
                os.path.join(prev_task_path, 'results'))
            _link_old_models(work_path, old_model_files, ii)
    # Copy user defined forward files
    symlink_user_forward_files(mdata=mdata, task_type="train", work_path=work_path)


def run_train(iter_index,
              jdata,
              mdata):
    numb_models = jdata['numb_models']
    train_input_file = default_train_input_file
    training_reuse_iter = jdata.get('training_reuse_iter')
    training_init_model = jdata.get('training_init_model', False)
    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        training_init_model = True
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    
    # train_command need to specified for nequip
    train_command = mdata.get('train_command', 'config_energy_force')
    train_resources = mdata['train_resources']

    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, 'copied')
    if os.path.isfile(copy_flag):
        log_task('copied model, do not train')
        return 
    # make tasks
    all_tasks = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        all_tasks.append(task_path)
    commands = []
    # need add command for nequip; for nequip, we need to give the train num
    
    n_train = 0
    init_data_sys_ = jdata['init_data_sys']
    init_data_sys = []
    for ii in init_data_sys_ :
        init_data_sys.append(os.path.join('data.all/data.init', ii))
    trains_comm_data = []
    cwd = os.getcwd()
    os.chdir(work_path)
    fp_data = glob.glob(os.path.join('data.all/data.iters','iter.*','02.fp','data.*'))
    for ii in init_data_sys :
        trains_comm_data += glob.glob(os.path.join(ii, 'fp.hdf5'))
        n_train += h5py.File(os.path.join(ii, 'fp.hdf5'),'r')['_n_nodes'].shape[0]
    for ii in fp_data:
        trains_comm_data += glob.glob(os.path.join(ii, 'fp.hdf5'))
        n_train += h5py.File(os.path.join(ii, 'fp.hdf5'),'r')['_n_nodes'].shape[0]
    os.chdir(cwd)
    n_train = int(n_train)
    n_val = max(int(0.01 * n_train),64)
    n_train = n_train - n_val
    #config_spec = "{\'data_confi.n_train\':%s, \'data_config.n_val\': 0, \'data.config.path\':\'../data.all/:.fp.hdf5\'}" %(n_train)
    # may define the learning rate
    if n_train > 77500:
        epoch_sub = 2 
    else:
        epoch_sub = 1
    if LooseVersion(mdata["deepmd_version"]) >= LooseVersion('1') and LooseVersion(mdata["deepmd_version"]) < LooseVersion('3'):
        if training_init_model:
            command = "python3 train.py --config %s --config_spec \"{'data_config.n_train':%s,'data_config.n_val':%s,'data_config.path':'../data.all/:.+fp.hdf5','batch_size':64,'stride':%s, 'epoch_subdivision':%s, 'md':False, 'early_stopping_delta':{'training_loss':0.5},'learning_rate':0.003,'metric_key':'validation_loss','early_stopping_patiences':{'training_loss':20}}\" --resume_from old/results/default_project/default/last.pt" %(train_command, n_train, n_val, max(int(n_train/10000),1),epoch_sub)
        else:
            #command = "python3 train.py --config %s --config_spec \"{'data_config.n_train':%s,'data_config.n_val':%s,'data_config.path':'../data.all/:.+fp.hdf5','batch_size':64,'stride':%s, 'epoch_subdivision':%s, 'md':False, 'early_stopping_delta':{'training_loss':0.5},'learning_rate':0.01,'metric_key':'validation_loss','early_stopping_patiences':{'training_loss':20}}\"" %(train_command, n_train, n_val, max(int(n_train/10000),1),epoch_sub)
            command = "python3 train.py --config %s --config_spec \"{'data_config.n_train':%s,'data_config.n_val':%s,'data_config.path':'../data.all/:.+fp.hdf5','batch_size':64,'stride':%s, 'epoch_subdivision':%s, 'md':False, 'early_stopping_delta':{'training_loss':0.5},'learning_rate':0.008,'metric_key':'validation_loss','early_stopping_patiences':{'training_loss':20}}\" --resume_from old/results/default_project/default/last.pt" %(train_command, n_train, n_val, max(int(n_train/10000),1),epoch_sub)
        commands.append(command)
    else:
        raise RuntimeError("DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!" )

    run_tasks = [os.path.basename(ii) for ii in all_tasks]
    forward_files = ['train.py']
    #if training_init_model:
    #if iter_index > 0:
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'trainer.pt')]
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'last.pt')]
    forward_files += [os.path.join('old','results', 'default_project', 'default', 'best.pt')]
    backward_files = ['results']

    try:
        train_group_size = 1
        mdata['train_resources']['group_size'] = 1
    except:
        train_group_size = 1
        mdata['train_resources']['group_size'] = 1

    api_version = mdata.get('api_version', '0.9')

    user_forward_files = mdata.get("train" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("train" + "_user_backward_files", [])

    if LooseVersion(api_version) < LooseVersion('1.0'):
        warnings.warn(f"the dpdispatcher will be updated to new version."
            f"And the interface may be changed. Please check the documents for more details")
        dispatcher = make_dispatcher(mdata['train_machine'], mdata['train_resources'], work_path, run_tasks, train_group_size)
        dispatcher.run_jobs(mdata['train_resources'],
                        commands,
                        work_path,
                        run_tasks,
                        train_group_size,
                        trans_comm_data,
                        forward_files,
                        backward_files,
                        outlog = 'train.log',
                        errlog = 'train.log')
    elif LooseVersion(api_version) >= LooseVersion('1.0'):
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


def post_train (iter_index, 
                jdata,
                mdata) :
        # load json param
    generalML = jdata.get('generalML',True)
    numb_models = jdata['numb_models']
    nequip_md = jdata.get('nequip_md',True)
    xtb_md = jdata.get('xtb_md',True)
    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, 'copied')
    if os.path.isfile(copy_flag) :
        log_task('copied model, do not post train')
        return
    # symlink models either dp or nequip
    for ii in range(numb_models) :
        if not jdata.get("nequip", False):
            model_name = 'frozen_model.pb'
        else:
            model_name = 'results/default_project/default/best.pt'
        task_file = os.path.join(train_task_fmt % ii, model_name)
        if not jdata.get("nequip",False):
            ofile = os.path.join(work_path, 'graph.%03d.pb' % ii)
        else:
            ofile = os.path.join(work_path, 'best.%03d.pt' % ii) 
        if os.path.isfile(ofile) :
            os.remove(ofile)
        os.symlink(task_file, ofile)
        if generalML == True and ii == 0:
            if nequip_md == True:
                task_general_file = os.path.join('./generalpt','general_graph.00'+str(ii)+'.pt')
                task_general_file = os.path.abspath(task_general_file)
                general_ofile = os.path.join(work_path,'general_graph.%03d.pt' % ii)
                os.symlink(task_general_file,general_ofile)
            elif xtb_md == True:
                pass
            else:
                task_general_file = os.path.join('./generalpb','general_graph.00'+str(ii)+'.pb')
                task_general_file = os.path.abspath(task_general_file)
                general_ofile = os.path.join(work_path,'general_graph.%03d.pb' % ii)
                os.symlink(task_general_file,general_ofile)


def make_model_devi(iter_index,
                   jdata,
                   mdata) :
    model_devi_engine = jdata.get('model_devi_engine', 'lammps')
    model_devi_jobs = jdata['model_devi_jobs']
    generalML = jdata.get('generalML',True)
    nequip_md = jdata.get('nequip_md',True)
    xtb_md = jdata.get('xtb_md',True)
    if (iter_index >= len(model_devi_jobs)) :
        return False
    cur_job = model_devi_jobs[iter_index]
    if "sys_configs_prefix" in jdata:
        sys_configs = [] 
        for sys_list in jdata["sys_configs"]:
            #assert (isinstance(sys_list, list) ), "Currently only support type list for sys in 'sys_conifgs' "
            temp_sys_list = [os.path.join(jdata["sys_configs_prefix"], sys) for sys in sys_list]
            sys_configs.append(temp_sys_list)
    else:
        sys_configs = jdata['sys_configs']

    # whether using orgin configs
    shuffle_poscar = jdata['shuffle_poscar']

    sys_idx = expand_idx(cur_job['sys_idx'])
    if (len(sys_idx) != len(list(set(sys_idx)))) :
        raise RuntimeError("system index should be uniq")
    # add cur_dir as the directory of topo.txt and atompair_length.txt
    conf_systems = []; conf_dirs = []
    for idx in sys_idx : 
        cur_systems = []; cur_dir = []
        ss = sys_configs[idx]
        for ii in ss :
            cur_systems += glob.glob(ii)
        cur_systems.sort()
        cur_systems = [os.path.abspath(ii) for ii in cur_systems]
        for ii in cur_systems:
            f_name = os.path.basename(ii)
            cur_dir.append(ii[:-len(f_name)])
        conf_systems.append (cur_systems); conf_dirs.append(cur_dir)
    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    if not jdata.get("nequip", False):
        models = sorted(glob.glob(os.path.join(train_path, "graph*pb")))
    else:
        models = sorted(glob.glob(os.path.join(train_path, "best*pt")))
    # !!!!!!!!! modify part
    if generalML == True:
        if nequip_md == True:
            models1 = sorted(glob.glob(os.path.join(train_path, "general_graph*pt")))
        elif xtb_md == True:
            models1 = []
        else:
            models1 = sorted(glob.glob(os.path.join(train_path, "general_graph*pb")))
    work_path = os.path.join(iter_name, model_devi_name)
    create_path(work_path)
    for mm in models :
        model_name = os.path.basename(mm)
        os.symlink(mm, os.path.join(work_path, model_name))
    if generalML == True:
        for mm1 in models1:
            model_name = os.path.basename(mm1)
            os.symlink(mm1, os.path.join(work_path, model_name))
    # add inference.py to 01.model_devi
    #cwd = os.getcwd()
    #os.symlink(os.path.join(cwd,'inference.py'),os.path.join(work_path,'inference.py'))
    with open(os.path.join(work_path, 'cur_job.json'), 'w') as outfile:
        json.dump(cur_job, outfile, indent = 4)

    conf_path = os.path.join(work_path, 'confs')
    create_path(conf_path)
    sys_counter = 0 

    # also symlink the topo and bond information
    for ss,dirs in zip(conf_systems,conf_dirs):
        conf_counter = 0
        for cc,dir0 in zip(ss,dirs) :
            if model_devi_engine == "lammps":
                conf_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
                orig_poscar_name = conf_name + '.orig.poscar'
                poscar_name = conf_name + '.poscar'
                lmp_name = conf_name + '.lmp'
                topo_name = conf_name + '.topo'
                bondlength_name = conf_name + '.bondlength'

                if shuffle_poscar :
                    os.symlink(cc, os.path.join(conf_path, orig_poscar_name))
                    poscar_shuffle(os.path.join(conf_path, orig_poscar_name),
                                   os.path.join(conf_path, poscar_name))
                else :
                    os.symlink(cc, os.path.join(conf_path, poscar_name))
                    os.symlink(os.path.join(dir0,'topo.txt'),os.path.join(conf_path,topo_name))
                    os.symlink(os.path.abspath('./atompair_length.txt'),os.path.join(conf_path,bondlength_name))

                if 'sys_format' in jdata:
                    fmt = jdata['sys_format']
                else:
                    fmt = 'vasp/poscar'
                if fmt == 'gaussian/log':
                    system = dpdata.LabeledSystem(os.path.join(conf_path, poscar_name), fmt = fmt, type_map = jdata['type_map'])
                else:
                    system = dpdata.System(os.path.join(conf_path, poscar_name), fmt = fmt, type_map = jdata['type_map'])
                if jdata.get('model_devi_nopbc', False):
                    system.remove_pbc()
                system.to_lammps_lmp(os.path.join(conf_path, lmp_name))
            elif model_devi_engine == "gromacs":
                pass
            conf_counter += 1
        sys_counter += 1

    input_mode = "native"
    if "template" in cur_job:
        input_mode = "revise_template"
    use_plm = jdata.get('model_devi_plumed', False)
    use_plm_path = jdata.get('model_devi_plumed_path', False)
    if input_mode == "native":
        if model_devi_engine == "lammps":
            _make_model_devi_native(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "gromacs":
            _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems)
        else:
            raise RuntimeError("unknown model_devi engine", model_devi_engine)
    elif input_mode == "revise_template":
        _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems)
    else:
        raise RuntimeError('unknown model_devi input mode', input_mode)
    #Copy user defined forward_files
    symlink_user_forward_files(mdata=mdata, task_type="model_devi", work_path=work_path)
    return True

def _make_model_devi_native(iter_index, jdata, mdata, conf_systems):
    model_devi_jobs = jdata['model_devi_jobs']
    generalML = jdata.get('generalML',True)
    nequip_md = jdata.get('nequip_md',True)
    xtb_md = jdata.get('xtb_md',True)
    if (iter_index >= len(model_devi_jobs)) :
        return False
    cur_job = model_devi_jobs[iter_index]
    ensemble, nsteps, trj_freq, temps, press, pka_e, dt = parse_cur_job(cur_job)
    if dt is not None :
        model_devi_dt = dt
    sys_idx = expand_idx(cur_job['sys_idx'])
    if (len(sys_idx) != len(list(set(sys_idx)))) :
        raise RuntimeError("system index should be uniq")

    use_ele_temp = jdata.get('use_ele_temp', 0)
    model_devi_dt = jdata['model_devi_dt']
    model_devi_neidelay = None
    if 'model_devi_neidelay' in jdata :
        model_devi_neidelay = jdata['model_devi_neidelay']
    model_devi_taut = 0.1
    if 'model_devi_taut' in jdata :
        model_devi_taut = jdata['model_devi_taut']
    model_devi_taup = 0.5
    if 'model_devi_taup' in jdata :
        model_devi_taup = jdata['model_devi_taup']
    mass_map = jdata['mass_map']
    nopbc = jdata.get('model_devi_nopbc', False)
    
    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    models = glob.glob(os.path.join(train_path, "graph*pb"))
    if generalML == True:
        if nequip_md == True:
            models = glob.glob(os.path.join(train_path, "general_graph*pt"))
        elif xtb_md == True:
            models = []
        else:
            models = glob.glob(os.path.join(train_path, "general_graph*pb"))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join('..', os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)

    sys_counter = 0 
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss :
            for tt_ in temps:
                if use_ele_temp:
                    if type(tt_) == list:
                        tt = tt_[0]
                        if use_ele_temp == 1:
                            te_f = tt_[1]
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt_[1]
                    else:
                        assert(type(tt_) == float or type(tt_) == int)
                        tt = float(tt_)
                        if use_ele_temp == 1:
                            te_f = tt
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt
                else :
                    tt = tt_
                    te_f = None
                    te_a = None
                for pp in press:
                    task_name = make_model_devi_task_name(sys_idx[sys_counter], task_counter)
                    conf_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter) + '.lmp'
                    # symlink the topo file and the bondlength file
                    topo_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter) + '.topo'
                    bondlength_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter) + '.bondlength'

                    task_path = os.path.join(work_path, task_name)
                    # dlog.info(task_path)
                    create_path(task_path)
                    create_path(os.path.join(task_path, 'traj'))
                    loc_conf_name = 'conf.lmp'
                    os.symlink(os.path.join(os.path.join('..','confs'), conf_name),
                               os.path.join(task_path, loc_conf_name) )
                    os.symlink(os.path.join(os.path.join('..','confs'), topo_name),
                               os.path.join(task_path, 'conf.topo') )
                    os.symlink(os.path.join(os.path.join('..','confs'), bondlength_name), 
                               os.path.join(task_path, 'conf.bondlength'))
                    cwd_ = os.getcwd()
                    if not os.path.isfile(os.path.join(task_path,'inference.py')):
                        os.symlink(os.path.join(cwd_,'inference.py'),os.path.join(task_path,'inference.py'))
                        os.symlink(os.path.join(cwd_,'topo.py'),os.path.join(task_path,'topo.py'))
                        if nequip_md == True:
                            os.symlink(os.path.join(cwd_,'e3_layer_md.py'),os.path.join(task_path,'e3_layer_md.py'))
                        if xtb_md == True:
                            os.symlink(os.path.join(cwd_,'xtb_md.py'),os.path.join(task_path,'xtb_md.py'))
                    os.chdir(task_path)
                    try:
                        mdata["deepmd_version"]
                    except KeyError:
                        mdata = set_version(mdata)
                    deepmd_version = mdata['deepmd_version']
                    max_seed_list = np.random.rand(iter_index+1)*100000000
                    max_seed = int(str(int(max_seed_list[-1]))[-6:])
                    file_c = make_lammps_input(ensemble,
                                               loc_conf_name,
                                               task_model_list,
                                               nsteps,
                                               model_devi_dt,
                                               model_devi_neidelay,
                                               trj_freq,
                                               mass_map,
                                               tt,
                                               jdata = jdata,
                                               tau_t = model_devi_taut,
                                               pres = pp,
                                               tau_p = model_devi_taup,
                                               pka_e = pka_e,
                                               ele_temp_f = te_f,
                                               ele_temp_a = te_a,
                                               max_seed = max_seed,
                                               nopbc = nopbc,
                                               deepmd_version = deepmd_version)
                    job = {}
                    job["ensemble"] = ensemble
                    job["press"] = pp
                    job["temps"] = tt
                    if te_f is not None:
                        job["ele_temp"] = te_f
                    if te_a is not None:
                        job["ele_temp"] = te_a
                    job["model_devi_dt"] =  model_devi_dt
                    with open('job.json', 'w') as _outfile:
                        json.dump(job, _outfile, indent = 4)
                    with open('q_net.txt','w') as fp :
                        q_net = jdata['charge_net'][sys_idx[sys_counter]]
                        fp.write('%s'%(str(q_net))) 
                    os.chdir(cwd_)
                    with open(os.path.join(task_path, 'input.lammps'), 'w') as fp :
                        fp.write(file_c)
                    task_counter += 1
            conf_counter += 1
        sys_counter += 1

def run_model_devi (iter_index, 
                    jdata,
                    mdata):
    # !!!!!!! we also need nequip to perfrom md later
    generalML = jdata.get('generalML',True)
    nequipMD = jdata.get('nequip_md',False)
    xtbmd = jdata.get('xtb_md',False)
    model_devi_exec = mdata['model_devi_command']

    model_devi_group_size = mdata['model_devi_group_size']
    model_devi_resources = mdata['model_devi_resources']
    use_plm = jdata.get('model_devi_plumed', False)
    use_plm_path = jdata.get('model_devi_plumed_path', False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, model_devi_name)
    assert(os.path.isdir(work_path))

    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    fp = open (os.path.join(work_path, 'cur_job.json'), 'r')
    cur_job = json.load (fp)

    run_tasks_ = all_task
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    if nequipMD == True:
        if generalML == True:
            all_models = glob.glob(os.path.join(work_path, 'general_graph*pt'))
            model_names = [os.path.basename(ii) for ii in all_models]
        else:
            all_models = glob.glob(os.path.join(work_path, 'best*.pt'))
            model_names = [os.path.basename(ii) for ii in all_models]
    else:
        all_models = glob.glob(os.path.join(work_path, 'general_graph*pb'))
        model_names = [os.path.basename(ii) for ii in all_models]

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine == "lammps":
        if nequipMD == True:
            if generalML == True:        
                # this function is seldom to be used, i will modify later
                command = "python3 e3_layer_md.py %s %s %s" %(cur_job['temps'][-1],cur_job['nsteps']+cur_job['trj_freq'], cur_job['trj_freq']) 
                command = "/bin/sh -c '%s'" % command
                commands = [command]
                forward_files = ['conf.lmp', 'e3_layer_md.py','traj']
                backward_files = ['monitor.csv','model_devi.log','traj']
            else:
                command = "python3 e3_layer_md.py %s %s %s %s %s 1" %(cur_job['temps'][-1],cur_job['nsteps']+cur_job['trj_freq'], cur_job['trj_freq'],jdata['bond_hi'],jdata['bond_lo']) 
                command = "/bin/sh -c '%s'" % command
                commands = [command]
                forward_files = ['conf.lmp', 'e3_layer_md.py', 'topo.py', 'inference.py', 'conf.topo', 'conf.bondlength'] 
                backward_files = ['monitor.csv','model_devi.log','traj.hdf5','f_pred0.hdf5','f_pred1.hdf5','f_pred2.hdf5','f_pred3.hdf5','reasonable.txt','model_devi_online.out']
        elif xtbmd == True:
            # this function will be modify in the futurew
            command = "python3 xtb_md.py %s %s %s" %(cur_job['temps'][-1],(cur_job['nsteps']+cur_job['trj_freq'])/1000.,cur_job['trj_freq'])
            command = "/bin/sh -c '%s'" % command
            commands = [command]
            forward_files = ['conf.lmp', 'xtb_md.py', 'q_net.txt','traj']
            backward_files = ['traj']
        else:
            command = "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }" % (model_devi_exec, model_devi_exec)
            command = "/bin/sh -c '%s'" % command
            commands = [command]
            forward_files = ['conf.lmp', 'input.lammps', 'traj']
            backward_files = ['model_devi.out', 'model_devi.log', 'traj']
            if use_plm:
                forward_files += ['input.plumed']
                # backward_files += ['output.plumed']
                backward_files += ['output.plumed','COLVAR']
                if use_plm_path:
                    forward_files += ['plmpath.pdb']

    cwd = os.getcwd()

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    api_version = mdata.get('api_version', '0.9')
 
    if LooseVersion(api_version) < LooseVersion('1.0'):
        warnings.warn(f"the dpdispatcher will be updated to new version."
            f"And the interface may be changed. Please check the documents for more details")
        dispatcher = make_dispatcher(mdata['model_devi_machine'], mdata['model_devi_resources'], work_path, run_tasks, model_devi_group_size)
        dispatcher.run_jobs(mdata['model_devi_resources'],
                        commands,
                        work_path,
                        run_tasks,
                        model_devi_group_size,
                        model_names,
                        forward_files,
                        backward_files,
                        outlog = 'model_devi.log',
                        errlog = 'model_devi.log')

    elif LooseVersion(api_version) >= LooseVersion('1.0'):
        submission = make_submission(
            mdata['model_devi_machine'],
            mdata['model_devi_resources'],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog = 'model_devi.log',
            errlog = 'model_devi.log')
        submission.run_submission()

def copy_model_devi (iter_index,
                     jdata,
                     mdata) :
    # !!!!!!! this operation can aviod too much md simulation, only model_deviatoin calculations are needed 
    generalML = jdata.get('generalML',True)
    nequipMD = jdata.get('nequip_md',False)
    xtbmd = jdata.get('xtb_md',False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, model_devi_name)
    assert(os.path.isdir(work_path))

    model_devi_group_size = mdata['model_devi_group_size']
    model_devi_resources = mdata['model_devi_resources']

    fp = open (os.path.join(work_path, 'cur_job.json'), 'r')
    cur_job = json.load (fp)

    prev_name = make_iter_name(iter_index - 1)
    prev_work_path = os.path.join(prev_name, model_devi_name)

    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    run_tasks_ = all_task
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]

    if nequipMD == True:
        if generalML == True:
            all_models = glob.glob(os.path.join(work_path, 'general_graph*pt'))
            model_names = [os.path.basename(ii) for ii in all_models]
        else:
            all_models = glob.glob(os.path.join(work_path, 'best*.pt'))
            model_names = [os.path.basename(ii) for ii in all_models]
    else:
        all_models = glob.glob(os.path.join(work_path, 'general_graph*pb'))
        model_names = [os.path.basename(ii) for ii in all_models]

    for task in run_tasks:
        os.symlink(os.path.abspath(os.path.join(prev_work_path, task, 'traj.hdf5')), os.path.join(work_path, task, 'traj_old.hdf5'))
        os.symlink(os.path.abspath(os.path.join(prev_work_path, task, 'reasonable.txt')), os.path.join(work_path, task, 'reasonable_old.txt'))
    command = "python3 e3_layer_md.py %s %s %s %s %s %s" %(cur_job['temps'][-1],cur_job['nsteps']+cur_job['trj_freq'], cur_job['trj_freq'],jdata['bond_hi'],jdata['bond_lo'], str(0))
    command = "/bin/sh -c '%s'" % command
    commands = [command]
    forward_files = ['conf.lmp', 'e3_layer_md.py', 'topo.py', 'inference.py', 'conf.topo', 'conf.bondlength', 'traj_old.hdf5', 'reasonable_old.txt']
    backward_files = ['model_devi.log','f_pred0.hdf5','f_pred1.hdf5','f_pred2.hdf5','f_pred3.hdf5','reasonable.txt','model_devi_online.out','traj.hdf5']
    
    cwd = os.getcwd()

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    api_version = mdata.get('api_version', '0.9')

    if LooseVersion(api_version) < LooseVersion('1.0'):
        warnings.warn(f"the dpdispatcher will be updated to new version."
            f"And the interface may be changed. Please check the documents for more details")
        dispatcher = make_dispatcher(mdata['model_devi_machine'], mdata['model_devi_resources'], work_path, run_tasks, model_devi_group_size)
        dispatcher.run_jobs(mdata['model_devi_resources'],
                        commands,
                        work_path,
                        run_tasks,
                        model_devi_group_size,
                        model_names,
                        forward_files,
                        backward_files,
                        outlog = 'model_devi.log',
                        errlog = 'model_devi.log')

    elif LooseVersion(api_version) >= LooseVersion('1.0'):
        submission = make_submission(
            mdata['model_devi_machine'],
            mdata['model_devi_resources'],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog = 'model_devi.log',
            errlog = 'model_devi.log')
        submission.run_submission()

def post_model_devi (iter_index, 
                     jdata,
                     mdata) :
    # generate model_deviation with dp model
    generalML = jdata.get('generalML',True)
    if generalML == True:
       model_devi_exec = mdata['model_devi_command']
       model_devi_group_size = mdata['model_devi_group_size']
       model_devi_resources = mdata['model_devi_resources']
       use_plm = jdata.get('model_devi_plumed', False)
       use_plm_path = jdata.get('model_devi_plumed_path', False)
       type_map = jdata.get('type_map')
       iter_name = make_iter_name(iter_index)
       work_path = os.path.join(iter_name,model_devi_name)
       assert(os.path.isdir(work_path))

       # generate command
       all_task = glob.glob(os.path.join(work_path, "task.*"))
       all_task.sort()
       fp = open (os.path.join(work_path, 'cur_job.json'),'r')
       cur_job = json.load(fp)

       run_tasks_ = all_task
       run_tasks = [os.path.basename(ii) for ii in run_tasks_]
       # get models
       all_models = glob.glob(os.path.join(work_path, "best*pt"))
       model_names = [os.path.basename(ii) for ii in all_models]
       trj_freq = cur_job.get("trj_freq",10); n_frames = int(cur_job.get("nsteps",1000)/trj_freq)+1
       attrs =  {'pos':('node','1x1o'),'species': ('node','1x0e'), 'energy': ('graph', '1x0e'), 'forces': ('node', '1x1o')}
       
       command = "python3 topo.py %s %s %s" %(jdata["bond_hi"],jdata["bond_lo"],trj_freq) 
       command += f"&& python3 -c \"import numpy as np; from e3_layers.data import Batch; import ase; atomic_n = ase.atom.atomic_numbers; coord = np.load('traj_deepmd/set.000/coord.npy'); coord = np.array(coord,dtype=np.single); type = np.loadtxt('traj_deepmd/type.raw'); type_map = {type_map}; species_n = [atomic_n[type_map[int(u)]] for u in type]; species_n = np.array(species_n,dtype=np.intc); e = np.array(0., dtype=np.single); lst = []; [lst.append(dict(pos=coord[ii].reshape((len(species_n),3)),energy=e, forces = coord[ii].reshape((len(species_n),3)), species=species_n)) for ii in range(len(coord))]; path = 'traj.hdf5'; attrs = {attrs}; batch = Batch.from_data_list(lst, attrs); batch.dumpHDF5(path)\""
       command += "&& python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.000.pt --output_keys forces --output_path f_pred0.hdf5" 
       command += "&& python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.001.pt --output_keys forces --output_path f_pred1.hdf5"
       command += "&& python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.002.pt --output_keys forces --output_path f_pred2.hdf5"
       command += "&& python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.003.pt --output_keys forces --output_path f_pred3.hdf5"
       
       commands = [command] 
       forward_files = ['traj']
       forward_files += ['inference.py','conf.topo','conf.bondlength','topo.py','conf.lmp']
       backward_files = ['f_pred0.hdf5','f_pred1.hdf5','f_pred2.hdf5','f_pred3.hdf5','traj.hdf5','reasonable.txt']
       cwd = os.getcwd()
       user_forward_files = mdata.get("model_devi" + "_user_forward_files",[])
       forward_files += [os.path.basename(file) for file in user_forward_files]
       backward_files += mdata.get("model_devi" + "_user_backward_files", [])
       api_version = mdata.get('api_version', '0.9')
       mpreddata = mdata['train_resources']
       mpreddata['source_list'] = ['/root/e3_layer.sh']
       mpreddata['number_node'] = mpreddata['number_node'] - 49
       mpreddata['group_size'] = 10 #mdata['model_devi_resources']['group_size'] 
       if LooseVersion(api_version) < LooseVersion('1.0'):
          warnings.warn(f"the dpdispatcher will be updated to new version."
            f"And the interface may be changed. Please check the documents for more details")
          dispatcher = make_dispatcher(mdata['train_machine'], mpreddata, work_path, run_tasks, model_devi_group_size)
          dispatcher.run_jobs(mpreddata,
                        commands,
                        work_path,
                        run_tasks,
                        model_devi_group_size,
                        model_names,
                        forward_files,
                        backward_files,
                        outlog = 'model_devi.log',
                        errlog = 'model_devi.log')

       elif LooseVersion(api_version) >= LooseVersion('1.0'):
           submission = make_submission(
            mdata['train_machine'],
            mpreddata,
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog = 'model_devi.log',
            errlog = 'model_devi.log')
       submission.run_submission()
       all_tasks = glob.glob(os.path.join(work_path, "task.*"))
       process_model_devi(all_tasks,trj_freq,'model_devi_online.out')
    else:
        pass


def _read_model_devi_file(
        task_path : str, 
        model_devi_f_avg_relative : bool = False, 
        generalML = True
):
    if generalML == True:
        model_devi = np.loadtxt(os.path.join(task_path, 'model_devi_online.out'))
    else:
        model_devi = np.loadtxt(os.path.join(task_path, 'model_devi.out'))
    if model_devi_f_avg_relative :
        trajs = glob.glob(os.path.join(task_path, 'traj', '*.lammpstrj'))
        all_f = []
        for ii in trajs:
            all_f.append(get_dumped_forces(ii))
        all_f = np.array(all_f)
        all_f = all_f.reshape([-1,3])
        avg_f = np.sqrt(np.average(np.sum(np.square(all_f), axis = 1)))
        model_devi[:,4:7] = model_devi[:,4:7] / avg_f
        np.savetxt(os.path.join(task_path, 'model_devi_avgf.out'), model_devi, fmt='%16.6e')
    return model_devi 

def _select_by_model_devi_adaptive_trust_low(
        modd_system_task: List[str],
        f_trust_lo : float, 
        f_trust_hi : float,
        numb_candi_f : int,
        perc_candi_f : float,
        v_trust_hi : float,
        numb_candi_v : int,
        perc_candi_v : float,
        model_devi_skip : int = 0,
        model_devi_f_avg_relative : bool = False,
        generalML : bool = False):
    """
    modd_system_task    model deviation tasks belonging to one system
    f_trust_hi
    numb_candi_f        number of candidate due to the f model deviation
    perc_candi_f        percentage of candidate due to the f model deviation
    v_trust_hi
    numb_candi_v        number of candidate due to the v model deviation
    perc_candi_v        percentage of candidate due to the v model deviation
    model_devi_skip

    returns
    accur               the accurate set
    candi               the candidate set
    failed              the failed set
    counter             counters, number of elements in the sets
    f_trust_lo          adapted trust level of f
    v_trust_low         adapted trust level of v
    """
    idx_v = 1
    idx_f = 4
    accur = set()
    candi = set()
    failed = []
    coll_v = []
    coll_f = []
    coll_f_1 = []
    for tt in modd_system_task:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_devi = np.loadtxt(os.path.join(tt, 'model_devi_online.out'))
            #!!!!! if unreasonable structres is observed, the structures behind them will be assumed as failed
            unreason = False
            for ii in range(model_devi.shape[0]):
                if model_devi[ii][0] < model_devi_skip :
                    continue
                cc = int(model_devi[ii][0])
                # tt: name of task folder
                # cc: time step of the frame
                md_v = model_devi[ii][idx_v]
                md_f = model_devi[ii][idx_f]
                if md_f > 900.:
                    unreason = True
                # if model_devi_f is frequently too high, the remain structure is unbelievable
                if generalML == False and md_f > f_trust_hi * 1.5 and md_f < 500.:
                    if ii < model_devi.shape[0] - 2:
                        if model_devi[ii+1][idx_f] > f_trust_hi * 1.5 and model_devi[ii+2][idx_f] > f_trust_hi * 1.5:
                            unreason = True
                if md_f > f_trust_hi or md_v > v_trust_hi or unreason == True:
                    failed.append([tt, cc])
                else:
                    coll_v.append([model_devi[ii][idx_v], tt, cc])
                    coll_f.append([model_devi[ii][idx_f], tt, cc])
                    # now accur takes all non-failed frames,
                    # will be substracted by candidate lat er
                    accur.add((tt, cc))
                    if md_f > f_trust_lo:
                        coll_f_1.append([model_devi[ii][idx_f], tt, cc])
    # sort
    coll_v.sort()
    coll_f.sort()
    coll_f_1.sort()
    assert(len(coll_v) == len(coll_f))

    # calcuate numbers
    numb_candi_v = max(numb_candi_v, int(perc_candi_v * 0.01 * len(coll_v)))
    numb_candi_f = max(numb_candi_f, int(perc_candi_f * 0.01 * len(coll_f)))

    # adjust number of candidate
    if len(coll_v) < numb_candi_v:
        numb_candi_v = len(coll_v)
    if len(coll_f_1) < numb_candi_f:
        numb_candi_f = len(coll_f_1)
    coll_val = np.array([condi[0] for condi in coll_f_1])
    # avoid unreasonable structure when MLFF is relatively good (the probability is small)
    numb_candi_unreason = 0
    if len(coll_f_1) > 0 and coll_f_1[-1][0] > max(coll_f_1[-numb_candi_f][0],f_trust_lo) * 3.5:
        numb_candi_unreason = min(len(np.where(coll_val > max(coll_f_1[-numb_candi_f][0],f_trust_lo) * 3.5)[0]),numb_candi_f-1)

    # compute trust lo
        # compute trust lo
    if numb_candi_v == 0:
        v_trust_lo = v_trust_hi
    else:
        v_trust_lo = coll_v[-numb_candi_v][0]
    if numb_candi_f == 0:
        f_trust_lo = f_trust_hi
    else:
        f_trust_lo = coll_f_1[-numb_candi_f][0]
    # add to candidate set
    for ii in range(len(coll_v) - numb_candi_v, len(coll_v)):
        candi.add(tuple(coll_v[ii][1:]))
    for ii in range(len(coll_f_1) - numb_candi_f, len(coll_f_1) - numb_candi_unreason):
        candi.add(tuple(coll_f_1[ii][1:]))
    
    # accurate set is substracted by the candidate set
    
    accur = accur - candi
    # convert to list
    candi = [list(ii) for ii in candi]
    accur = [list(ii) for ii in accur]
    # counters
    counter = Counter()
    counter['candidate'] = len(candi)
    counter['failed'] = len(failed)
    counter['accurate'] = len(accur)

    return accur, candi, failed, counter, f_trust_lo, v_trust_lo

def _make_fp_vasp_inner_loose (modd_path,
                         work_path,
                         prev_path,
                         model_devi_skip,
                         v_trust_lo,
                         v_trust_hi,
                         f_trust_lo,
                         f_trust_hi,
                         fp_task_min,
                         fp_task_max,
                         fp_link_files,
                         type_map,
                         jdata):
    """
    modd_path           string          path of model devi
    work_path           string          path of fp
    prev_path           string          path of previous fp
    fp_task_max         int             max number of tasks
    fp_link_files       [string]        linked files for fp, POTCAR for example
    fp_params           map             parameters for fp
    md_cond    recod the history of [do_md, part_fp] 
    do_md      whether do md or not 
    part_fp    whether only use part of candi 
    patience_num      times of md explore conv 
    largermse_num     times of accurate fp static
    unreason_num      losse_cri for unreasonable topo
    unreason_pre      whether reasonable 
    fp_hist_idx_all   all history cand
    fp_hist_idx_cur   present cand
    """

    modd_task = glob.glob(os.path.join(modd_path, "task.*"))
    modd_task.sort()
    system_index = []
    for ii in modd_task :
        system_index.append(os.path.basename(ii).split('.')[1])
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_tasks = []; numb_task_all = []; fp_candidate_all = []; fp_hist_idx_all = []; fail_num_all = []; task_num_all = []

    charges_recorder = [] # record charges for each fp_task, remember amons have differnt net charge
    charges_map = jdata.get("sys_charges", [])
    trj_freq = jdata["model_devi_jobs"][-1]["trj_freq"]
    cluster_cutoff = jdata['cluster_cutoff'] if jdata.get('use_clusters', False) else None
    model_devi_adapt_trust_lo = jdata.get('model_devi_adapt_trust_lo', False)
    model_devi_f_avg_relative = jdata.get('model_devi_f_avg_relative', False)
    # skip save *.out if detailed_report_make_fp is False, default is True
    detailed_report_make_fp = jdata.get("detailed_report_make_fp", True)
    # skip bad box criteria
    skip_bad_box = jdata.get('fp_skip_bad_box')
    # skip discrete structure in cluster
    fp_cluster_vacuum = jdata.get('fp_cluster_vacuum',None)
    generalML = jdata.get('generalML',True)
    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[int(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    # obtain the max tasks num for each config
    fp_task_max_acc_conv = int(len(jdata['all_sys_idx'])/len(jdata['model_devi_jobs'][-1]['sys_idx'])*jdata['fp_task_max'])
    fp_task_max_acc_conv = min(fp_task_max_acc_conv, jdata['fp_task_max_hi'])

    # static the fp tasks, if too much, only part of them do fp 
    total_candi_max = 0; total_fp_max = 0; do_md = True; do_part_fp = False
    if prev_path == None:
        do_md = True
    else:
        if os.path.isfile(os.path.join(prev_path,'md_cond')):
            do_md = np.loadtxt(os.path.join(prev_path,'md_cond'))[0]
            if do_md < 0.5:
                do_md = False
            else:
                do_md = True
        else:
            do_md = True
    
    for ss in system_index:
        modd_system_glob = os.path.join(modd_path, 'task.' + ss + '.*')
        modd_system_task = glob.glob(modd_system_glob)
        modd_system_task.sort()

        # convert global trust limitations to local ones
        f_trust_lo_sys = _trust_limitation_check(ss, f_trust_lo)
        f_trust_hi_sys = _trust_limitation_check(ss, f_trust_hi)
        v_trust_lo_sys = _trust_limitation_check(ss, v_trust_lo)
        v_trust_hi_sys = _trust_limitation_check(ss, v_trust_hi)

        # assumed e -> v
        numb_candi_f = jdata.get('model_devi_numb_candi_f', 10)
        numb_candi_v = jdata.get('model_devi_numb_candi_v', 0)
        perc_candi_f = jdata.get('model_devi_perc_candi_f', 0.)
        perc_candi_v = jdata.get('model_devi_perc_candi_v', 0.)
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter, f_trust_lo_ad, v_trust_lo_ad \
                =  _select_by_model_devi_adaptive_trust_low(
                    modd_system_task,f_trust_lo_sys,
                    f_trust_hi_sys, numb_candi_f, perc_candi_f,
                    v_trust_hi_sys, numb_candi_v, perc_candi_v,
                    model_devi_skip = model_devi_skip,
                    model_devi_f_avg_relative = model_devi_f_avg_relative,
                    generalML = generalML 
                )
        dlog.info("system {0:s} {1:9s} : f_trust_lo {2:6.3f}   v_trust_lo {3:6.3f}".format(ss, 'adapted', f_trust_lo_ad, v_trust_lo_ad))

        # print a report 
        if prev_path == None:
            patience_num = 0
            largermse_num = 0
            unreason_num = 0
            unreason_pre = 0 
            fp_hist_idx_all.append([])
        else:
            if os.path.isfile(os.path.join(prev_path,'md_cond')):
                if os.path.isfile(os.path.join(prev_path,str(ss)+'.fp_hist_idx')):
                    fp_hist_idx = np.loadtxt(os.path.join(prev_path,str(ss)+'.fp_hist_idx'))
                    if do_md == True:
                        fp_hist_idx = []
                else:
                    fp_hist_idx = []
                if isinstance(fp_hist_idx,list):
                    fp_hist_idx_all.append(fp_hist_idx)
                else:
                    if len(fp_hist_idx.shape) > 0: 
                        fp_hist_idx_all.append([int(idx) for idx in fp_hist_idx])
                    else:
                        fp_hist_idx_all.append([int(fp_hist_idx)])
            else:
                fp_hist_idx_all.append([])

            prev_static_path = os.path.join(prev_path,'static.'+ss)
            if os.path.isfile(prev_static_path):
                prev_static = np.loadtxt(prev_static_path)
                patience_num = int(prev_static[-1])
                largermse_num = int(prev_static[-2])
                # define the largermse_num, if rmse_f < rmse_f_cri_hi and conf_idx < 0.05 * conf_all
                if os.path.isfile(os.path.join(prev_path,'data.'+ss,'static.npz')):
                    rmse_f = np.load(os.path.join(prev_path,'data.'+ss,'static.npz'))['rmse_f']
                    if rmse_f < jdata['rmse_f_cri_hi']:
                        largermse_num += 1
                    else:
                        largermse_num = 0
                else:
                    if os.path.isfile(os.path.join(prev_path,'md_cond')):
                        part_fp = np.loadtxt(os.path.join(prev_path,'md_cond'))[1]
                        if int(part_fp) == 0:
                            largermse_num += 1
                    else:
                        largermse_num += 1
            else:
                patience_num = 0
                largermse_num = 0

            if os.path.isfile(os.path.join(prev_path,'unreason.'+ss)):
                unreason_num = np.loadtxt(os.path.join(prev_path,'unreason.'+ss))[0]
                unreason_pre = np.loadtxt(os.path.join(prev_path,'unreason.'+ss))[1]
            else:
                unreason_num = 0
                unreason_pre = 0
        
        fp_accurate_threshold = jdata.get('fp_accurate_threshold', 1)
        fp_accurate_soft_threshold = jdata.get('fp_accurate_soft_threshold', fp_accurate_threshold)

        fp_sum = sum(counter.values())
        task_num_all.append(fp_sum)
        # if md explore patience_num += 1
        if int(fp_task_max * (len(fp_rest_accurate)/fp_sum - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold)) == 0:
            patience_num += 1
        else:
            patience_num = 0
        
        # if conf_idx > 0.05 * conf_all or new temperature; largerrmse_num = 0
        if len(jdata["model_devi_jobs"]) > 1 and jdata["model_devi_jobs"][-2]["temps"][-1] < jdata["model_devi_jobs"][-1]["temps"][-1]:
            largermse_num = 0; patience_num = 0; unreason_num = 0
        if len(jdata['model_devi_jobs'][-1]['sys_idx'])/len(jdata['all_sys_idx']) > 1 -  jdata["conv_ratio"]:
            largermse_num = 0
        
        for cc_key, cc_value in counter.items():
            dlog.info("system {0:s} {1:9s} : {2:6d} in {3:6d} {4:6.2f} %".format(ss, cc_key, cc_value, fp_sum, cc_value/fp_sum*100))
        random.shuffle(fp_candidate)
        if detailed_report_make_fp:
            random.shuffle(fp_rest_failed)
            random.shuffle(fp_rest_accurate)
            with open(os.path.join(work_path,'candidate.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_candidate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(os.path.join(work_path,'rest_accurate.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_rest_accurate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(os.path.join(work_path,'rest_failed.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_rest_failed:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
        # set number of tasks
        accurate_ratio = float(counter['accurate']) / float(fp_sum)
        if accurate_ratio < fp_accurate_soft_threshold :
            this_fp_task_max = max(fp_task_max, fp_task_max_acc_conv)
        elif accurate_ratio >= fp_accurate_soft_threshold and accurate_ratio < fp_accurate_threshold:
            this_fp_task_max = int(fp_task_max * (accurate_ratio - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold))
            # if most of the configs are converged, enlarge the fp_tasks of rest configs 
            if this_fp_task_max > 0:
                this_fp_task_max = int(fp_task_max_acc_conv * (accurate_ratio - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold))
        else:
            this_fp_task_max = 0
        numb_task = min(this_fp_task_max, len(fp_candidate))
        
        # !!! add candidate and fp task here
        total_candi_max += len(fp_candidate) 
        total_fp_max += numb_task
        # !!! avoid too early stop md explore
        if len(fp_rest_failed)/fp_sum > 0.005:
            patience_num = 0
        
        f_reason = os.path.join(modd_path,'task.'+ss+'.000000','reasonable.txt')
        reason_idxs = np.loadtxt(f_reason); unreason_ratio = len(np.where(reason_idxs[:,0]<0.5)[0])/len(reason_idxs)
        # aviod unreason count from the same traj
        if do_md == False:
            pass
        else:
            if reason_idxs[-1][0] < 0.5 and unreason_ratio > 0.1:
                # record the md explore unreason times 
                if do_md == True:
                    unreason_num += 1
            else:
                unreason_num = 0 
        if unreason_ratio > 0.001:
            unreason_pre += 1
        np.savetxt(os.path.join(work_path,'static.'+ss),[len(fp_rest_accurate)/fp_sum,len(fp_candidate)/fp_sum,len(fp_rest_failed)/fp_sum,largermse_num,patience_num])
        np.savetxt(os.path.join(work_path,'unreason.'+ss),[unreason_num,unreason_pre]) 
        if (numb_task < fp_task_min):
            numb_task = 0
        dlog.info("system {0:s} accurate_ratio: {1:8.4f}    thresholds: {2:6.4f} and {3:6.4f}   eff. task min and max {4:4d} {5:4d}   number of fp tasks: {6:6d}".format(ss, accurate_ratio, fp_accurate_soft_threshold, fp_accurate_threshold, fp_task_min, this_fp_task_max, numb_task))
        numb_task_all.append(numb_task); fp_candidate_all.append(fp_candidate); fail_num_all.append(len(fp_rest_failed))
    
    #!!!! make fp tasks; here we need to sample part of fps if fps are too much; record the idx have chosen as the fp tasks
    if total_fp_max > jdata['total_fp_max']:
        do_part_fp = True
    else:
        do_part_fp = False
    
    np.savetxt(os.path.join(work_path,'md_cond'),[int(do_md),int(do_part_fp)])

    numb_task_all = np.array(numb_task_all)
    if do_part_fp == True:
        task_ratio = jdata['total_fp_max']/total_fp_max 
        _numb_task_all = [int(uu*task_ratio) for uu in numb_task_all]
        #!!!!! avoid all fp_num is 0
        if np.sum(np.array(_numb_task_all)) < jdata['total_fp_max'] * 0.5:
            _numb_task_all = np.zeros(len(numb_task_all))
            if len(_numb_task_all) > jdata['toal_fp_max']:
                _idx = np.argsort(-numb_task_all)[0:jdata['total_fp_max']]
                _numb_task_all[_idx] = 1.
            else:
                _numb_task_all = int(jdata['total_fp_max']/len(_numb_task_all))

            for ii,uu in enumerate(numb_task_all):
                _numb_task_all[ii] = int(min(_numb_task_all[ii],uu))
    else:
        _numb_task_all = numb_task_all
    
    for ss_idx, ss in enumerate(system_index):
        model_devi_engine = jdata.get("model_devi_engine", "lammps")
        count_bad_box = 0
        count_bad_cluster = 0
        numb_task = _numb_task_all[ss_idx]; fp_candidate = fp_candidate_all[ss_idx]; fp_hist_idx = fp_hist_idx_all[ss_idx]
        fp_hist_idx_cur = []; cc_remain = 0 # cc_remain means the idx exclude the history candidate
        for cc in range(numb_task) :
            tt = fp_candidate[cc][0]
            ii = fp_candidate[cc][1]
            ss = os.path.basename(tt).split('.')[1]
            conf_data = h5py.File(os.path.join(tt, "traj.hdf5"))
            coord = conf_data['pos'][:]; symbol = conf_data['species'][:]; n_frame = conf_data['energy'].shape[0]
            coord = coord.reshape(n_frame,int(len(coord)/n_frame),3); symbol = symbol.reshape(n_frame,int(len(symbol)/n_frame))

            if ii in fp_hist_idx:
                continue
            else:
                fp_hist_idx.append(ii)
                fp_hist_idx_cur.append(ii)

            pos = coord[int(ii/trj_freq)]; sym = symbol[0]
            job_name = os.path.join(tt, "job.json")
            job_name = os.path.abspath(job_name)

            fp_task_name = make_fp_task_name(int(ss), cc_remain)
            fp_task_path = os.path.join(work_path, fp_task_name)
            create_path(fp_task_path)
            fp_tasks.append(fp_task_path)
            if charges_map:
                charges_recorder.append(charges_map[int(ss)])
            cwd = os.getcwd()
            os.chdir(fp_task_path)
            dumpposcar(pos,sym,'POSCAR',type_map)
            cc_remain += 1 
            os.chdir(cwd)
        np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx_cur'),fp_hist_idx_cur)
        np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx'),fp_hist_idx)
        # static the history ratio; need modify
        if task_num_all[ss_idx] == fail_num_all[ss_idx]:
            cand_hist_ratio = len(fp_hist_idx)/task_num_all[ss_idx]
        else:
            cand_hist_ratio = len(fp_hist_idx)/(task_num_all[ss_idx] - fail_num_all[ss_idx])
        
        np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx_ratio'),[cand_hist_ratio])
        
        if count_bad_box > 0:
            dlog.info("system {0:s} skipped {1:6d} confs with bad box, {2:6d} remains".format(ss, count_bad_box, numb_task - count_bad_box))
        if count_bad_cluster > 0:
            dlog.info("system {0:s} skipped {1:6d} confs with bad cluster, {2:6d} remains".format(ss, count_bad_cluster, numb_task - count_bad_cluster))
###################################################################################################################################
    
    return fp_tasks

def _make_fp_vasp_inner (modd_path,
                         work_path,
                         prev_path,
                         model_devi_skip,
                         v_trust_lo,
                         v_trust_hi,
                         f_trust_lo,
                         f_trust_hi,
                         fp_task_min,
                         fp_task_max,
                         fp_link_files,
                         type_map,
                         jdata):
    """
    modd_path           string          path of model devi
    work_path           string          path of fp
    prev_path           string          path of previous fp
    fp_task_max         int             max number of tasks
    fp_link_files       [string]        linked files for fp, POTCAR for example
    fp_params           map             parameters for fp
    md_cond    recod the history of [do_md, part_fp] 
    do_md      whether do md or not 
    part_fp    whether only use part of candi 
    
    """

    modd_task = glob.glob(os.path.join(modd_path, "task.*"))
    modd_task.sort()
    system_index = []
    for ii in modd_task :
        system_index.append(os.path.basename(ii).split('.')[1])
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_tasks = []; numb_task_all = []; fp_candidate_all = []; fp_hist_idx_all = []

    charges_recorder = [] # record charges for each fp_task, remember amons have differnt net charge
    charges_map = jdata.get("sys_charges", [])

    cluster_cutoff = jdata['cluster_cutoff'] if jdata.get('use_clusters', False) else None
    model_devi_adapt_trust_lo = jdata.get('model_devi_adapt_trust_lo', False)
    model_devi_f_avg_relative = jdata.get('model_devi_f_avg_relative', False)
    # skip save *.out if detailed_report_make_fp is False, default is True
    detailed_report_make_fp = jdata.get("detailed_report_make_fp", True)
    # skip bad box criteria
    skip_bad_box = jdata.get('fp_skip_bad_box')
    # skip discrete structure in cluster
    fp_cluster_vacuum = jdata.get('fp_cluster_vacuum',None)
    generalML = jdata.get('generalML',True)
    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[int(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    fp_task_max_acc_conv = int(len(jdata['all_sys_idx'])/len(jdata['model_devi_jobs'][-1]['sys_idx'])*jdata['fp_task_max'])
    fp_task_max_acc_conv = min(fp_task_max_acc_conv, jdata['fp_task_max_hi'])

    # we need to static the number of whole fp candidate and fp task, as if over 10,000 configs will generate much too fp tasks
    total_candi_max = 0; total_fp_max = 0; do_md = True; do_part_fp = False
    if prev_path == None:
        last_do_md = 1
    else:
        if os.path.isfile(os.path.join(prev_path,'md_cond')):
            last_do_md = np.loadtxt(os.path.join(prev_path,'md_cond'))[0]
        else:
            last_do_md = 1
    for ss in system_index:
        modd_system_glob = os.path.join(modd_path, 'task.' + ss + '.*')
        modd_system_task = glob.glob(modd_system_glob)
        modd_system_task.sort()

        # convert global trust limitations to local ones
        f_trust_lo_sys = _trust_limitation_check(ss, f_trust_lo)
        f_trust_hi_sys = _trust_limitation_check(ss, f_trust_hi)
        v_trust_lo_sys = _trust_limitation_check(ss, v_trust_lo)
        v_trust_hi_sys = _trust_limitation_check(ss, v_trust_hi)

        # assumed e -> v
        numb_candi_f = jdata.get('model_devi_numb_candi_f', 10)
        numb_candi_v = jdata.get('model_devi_numb_candi_v', 0)
        perc_candi_f = jdata.get('model_devi_perc_candi_f', 0.)
        perc_candi_v = jdata.get('model_devi_perc_candi_v', 0.)
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter, f_trust_lo_ad, v_trust_lo_ad \
                =  _select_by_model_devi_adaptive_trust_low(
                    modd_system_task,f_trust_lo_sys,
                    f_trust_hi_sys, numb_candi_f, perc_candi_f,
                    v_trust_hi_sys, numb_candi_v, perc_candi_v,
                    model_devi_skip = model_devi_skip,
                    model_devi_f_avg_relative = model_devi_f_avg_relative,
                    generalML = generalML 
                )
        dlog.info("system {0:s} {1:9s} : f_trust_lo {2:6.3f}   v_trust_lo {3:6.3f}".format(ss, 'adapted', f_trust_lo_ad, v_trust_lo_ad))

        # print a report 
        if prev_path == None:
            patience_num = 0
            largermse_num = 0
            unreason_num = 0
            fp_hist_idx_all.append([])
        else:
            if os.path.isfile(os.path.join(prev_path,'md_cond')):
                last_do_md = np.loadtxt(os.path.join(prev_path,'md_cond'))[0]
                if os.path.isfile(os.path.join(prev_path,str(ss)+'.fp_hist_idx')):
                    fp_hist_idx = np.loadtxt(os.path.join(prev_path,str(ss)+'.fp_hist_idx'))
                    if int(last_do_md) == 1:
                        fp_hist_idx = []
                else:
                    fp_hist_idx = []
                if isinstance(fp_hist_idx,list):
                    fp_hist_idx_all.append(fp_hist_idx)
                else:
                    if len(fp_hist_idx.shape) > 0: 
                        fp_hist_idx_all.append([int(idx) for idx in fp_hist_idx])
                    else:
                        fp_hist_idx_all.append([int(fp_hist_idx)])
            else:
                fp_hist_idx_all.append([])

            prev_static_path = os.path.join(prev_path,'static.'+ss)
            if os.path.isfile(prev_static_path):
                prev_static = np.loadtxt(prev_static_path)
                patience_num = int(prev_static[-1])
                largermse_num = int(prev_static[-2])
                # define the largermse_num, if rmse_f < rmse_f_cri_hi and conf_idx < 0.05 * conf_all
                if os.path.isfile(os.path.join(prev_path,'data.'+ss,'static.npz')):
                    rmse_f = np.load(os.path.join(prev_path,'data.'+ss,'static.npz'))['rmse_f']
                    if rmse_f < jdata['rmse_f_cri_hi']:
                        largermse_num += 1
                    else:
                        largermse_num = 0
                else:
                    if os.path.isfile(os.path.join(prev_path,'md_cond')):
                        part_fp = np.loadtxt(os.path.join(prev_path,'md_cond'))[1]
                        if int(part_fp) == 0:
                            largermse_num += 1
                    else:
                        largermse_num += 1
            else:
                patience_num = 0
                largermse_num = 0

            if os.path.isfile(os.path.join(prev_path,'unreason.'+ss)):
                unreason_num = np.loadtxt(os.path.join(prev_path,'unreason.'+ss))
            else:
                unreason_num = 0
        
        fp_accurate_threshold = jdata.get('fp_accurate_threshold', 1)
        fp_accurate_soft_threshold = jdata.get('fp_accurate_soft_threshold', fp_accurate_threshold)

        fp_sum = sum(counter.values())

        if int(fp_task_max * (len(fp_rest_accurate)/fp_sum - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold)) == 0:
            # avoid not do md and the error is too small
            if last_do_md > 0.5:
                patience_num += 1
            else:
                # if rmse is small, even using previous md traj, we consider patience_num should add 1
                if largermse_num > 0:
                    patience_num += 1
        else:
            patience_num = 0
        
        # if conf_idx > 0.05 * conf_all or new temperature; largerrmse_num = 0
        if len(jdata["model_devi_jobs"]) > 1 and jdata["model_devi_jobs"][-2]["temps"][-1] < jdata["model_devi_jobs"][-1]["temps"][-1]:
            largermse_num = 0; patience_num = 0; unreason_num = 0
        if len(jdata['model_devi_jobs'][-1]['sys_idx'])/len(jdata['all_sys_idx']) > 1 -  jdata["conv_ratio"]:
            largermse_num = 0
        
        for cc_key, cc_value in counter.items():
            dlog.info("system {0:s} {1:9s} : {2:6d} in {3:6d} {4:6.2f} %".format(ss, cc_key, cc_value, fp_sum, cc_value/fp_sum*100))
        random.shuffle(fp_candidate)
        if detailed_report_make_fp:
            random.shuffle(fp_rest_failed)
            random.shuffle(fp_rest_accurate)
            with open(os.path.join(work_path,'candidate.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_candidate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(os.path.join(work_path,'rest_accurate.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_rest_accurate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(os.path.join(work_path,'rest_failed.shuffled.%s.out'%ss), 'w') as fp:
                for ii in fp_rest_failed:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
        # set number of tasks
        accurate_ratio = float(counter['accurate']) / float(fp_sum)
        if accurate_ratio < fp_accurate_soft_threshold :
            this_fp_task_max = max(fp_task_max, fp_task_max_acc_conv)
        elif accurate_ratio >= fp_accurate_soft_threshold and accurate_ratio < fp_accurate_threshold:
            this_fp_task_max = int(fp_task_max * (accurate_ratio - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold))
            # if most of the configs are converged, enlarge the fp_tasks of rest configs 
            if this_fp_task_max > 0:
                this_fp_task_max = int(fp_task_max_acc_conv * (accurate_ratio - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold))
        else:
            this_fp_task_max = 0
        numb_task = min(this_fp_task_max, len(fp_candidate))
        
        # !!! add candidate and fp task here
        total_candi_max += len(fp_candidate) 
        total_fp_max += numb_task
        # !!! avoid too early stop md explore
        if len(fp_rest_failed) > 0.005:
            patience_num = 0
         
        f_reason = os.path.join(modd_path,'task.'+ss+'.000000','reasonable.txt')
        reason_idxs = np.loadtxt(f_reason); unreason_ratio = len(np.where(reason_idxs[:,0]<0.5)[0])/len(reason_idxs)
        # aviod unreason count from the same traj
        if last_do_md < 0.5:
            pass
        else:
            if reason_idxs[-1][0] < 0.5 and unreason_ratio > 0.1:
                unreason_num += 1
            else:
                unreason_num = 0 
        np.savetxt(os.path.join(work_path,'static.'+ss),[len(fp_rest_accurate)/fp_sum,len(fp_candidate)/fp_sum,len(fp_rest_failed)/fp_sum,largermse_num,patience_num])
        np.savetxt(os.path.join(work_path,'unreason.'+ss),[unreason_num]) 
        if (numb_task < fp_task_min):
            numb_task = 0
        dlog.info("system {0:s} accurate_ratio: {1:8.4f}    thresholds: {2:6.4f} and {3:6.4f}   eff. task min and max {4:4d} {5:4d}   number of fp tasks: {6:6d}".format(ss, accurate_ratio, fp_accurate_soft_threshold, fp_accurate_threshold, fp_task_min, this_fp_task_max, numb_task))
        numb_task_all.append(numb_task); fp_candidate_all.append(fp_candidate)
    
    #!!!! make fp tasks; here we need to sample part of fps if fps are too much; record the idx have chosen as the fp tasks
    if total_candi_max > jdata['total_candi_min']:
        do_md = False
    else:
        do_md = True
    if total_fp_max > jdata['total_fp_max']:
        do_part_fp = True
    else:
        do_part_fp = False
    np.savetxt(os.path.join(work_path,'md_cond'),[int(do_md),int(do_part_fp)])
    numb_task_all = np.array(numb_task_all)

    if do_part_fp == True:
        task_ratio = jdata['total_fp_max']/total_fp_max 
        _numb_task_all = [int(uu*task_ratio) for uu in numb_task_all]
        #!!!!! avoid all fp_num is 0
        if np.sum(np.array(_numb_task_all)) < jdata['total_fp_max'] * 0.5:
            _numb_task_all = np.zeros(len(numb_task_all))
            if len(_numb_task_all) > jdata['toal_fp_max']:
                _idx = np.argsort(-numb_task_all)[0:jdata['total_fp_max']]
                _numb_task_all[_idx] = 1.
            else:
                _numb_task_all = int(jdata['total_fp_max']/len(_numb_task_all))

            for ii,uu in enumerate(numb_task_all):
                _numb_task_all[ii] = int(min(_numb_task_all[ii],uu))
    else:
        _numb_task_all = numb_task_all
    
    for ss_idx, ss in enumerate(system_index):
        model_devi_engine = jdata.get("model_devi_engine", "lammps")
        count_bad_box = 0
        count_bad_cluster = 0
        numb_task = _numb_task_all[ss_idx]; fp_candidate = fp_candidate_all[ss_idx]; fp_hist_idx = fp_hist_idx_all[ss_idx]
        fp_hist_idx_cur = []; cc_remain = 0 # cc_remain means the idx exclude the history candidate
        for cc in range(numb_task) :
            tt = fp_candidate[cc][0]
            ii = fp_candidate[cc][1]
            ss = os.path.basename(tt).split('.')[1]
################################################################################################################################
            conf_data = h5py.File(os.path.join(tt, "traj.hdf5"))
            coord = conf_data['pos'][:]; symbol = conf_data['species'][:]; n_frame = conf_data['energy'].shape[0]
            coord = coord.reshape(n_frame,int(len(coord)/n_frame),3); symbol = symbol.reshape(n_frame,int(len(symbol)/n_frame))

            if ii in fp_hist_idx:
                continue
            else:
                fp_hist_idx.append(ii)
                fp_hist_idx_cur.append(ii)

            pos = coord[ii]; sym = symbol[0]
            job_name = os.path.join(tt, "job.json")
            job_name = os.path.abspath(job_name)

            fp_task_name = make_fp_task_name(int(ss), cc_remain)
            fp_task_path = os.path.join(work_path, fp_task_name)
            create_path(fp_task_path)
            fp_tasks.append(fp_task_path)
            if charges_map:
                charges_recorder.append(charges_map[int(ss)])
            cwd = os.getcwd()
            os.chdir(fp_task_path)
            dumpposcar(pos,sym,'POSCAR',type_map)
            cc_remain += 1 
            os.chdir(cwd)
        np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx_cur'),fp_hist_idx_cur)
        if do_md == False:
            np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx'),fp_hist_idx)
        else:
            np.savetxt(os.path.join(work_path,str(ss)+'.fp_hist_idx'),[])
        if count_bad_box > 0:
            dlog.info("system {0:s} skipped {1:6d} confs with bad box, {2:6d} remains".format(ss, count_bad_box, numb_task - count_bad_box))
        if count_bad_cluster > 0:
            dlog.info("system {0:s} skipped {1:6d} confs with bad cluster, {2:6d} remains".format(ss, count_bad_cluster, numb_task - count_bad_cluster))
###################################################################################################################################
    # if actual fp tasks is much less than chosen fp tasks and fp_tasks is small
    if len(fp_tasks) < jdata['total_fp_max'] and len(fp_tasks) < 0.8 * np.sum(np.array(_numb_task_all)):
        do_md = True; do_part_fp = False

    return fp_tasks

def _make_fp_vasp_configs(iter_index, 
                         jdata):
    # !!! the criterion of trust_lo and trust_hi can optimize here
    fp_task_max = jdata['fp_task_max']
    model_devi_skip = jdata['model_devi_skip']
    v_trust_lo = jdata.get('model_devi_v_trust_lo', 1e10)
    v_trust_hi = jdata.get('model_devi_v_trust_hi', 1e10)
    f_trust_lo = jdata['model_devi_f_trust_lo']
    f_trust_hi = jdata['model_devi_f_trust_hi']
    type_map = jdata['type_map']
    iter_name = make_iter_name(iter_index)
    if iter_index > 0:
        previous_path = os.path.join(make_iter_name(iter_index-1),fp_name)
    else:
        previous_path = None
    work_path = os.path.join(iter_name, fp_name)
    create_path(work_path)

    modd_path = os.path.join(iter_name, model_devi_name)
    task_min = -1
    if os.path.isfile(os.path.join(modd_path, 'cur_job.json')) :
        cur_job = json.load(open(os.path.join(modd_path, 'cur_job.json'), 'r'))
        if 'task_min' in cur_job :
            task_min = cur_job['task_min']

    # make configs
    if jdata['generalML'] == True:
        fp_tasks = _make_fp_vasp_inner(modd_path, work_path,
                                   previous_path,
                                   model_devi_skip,
                                   v_trust_lo, v_trust_hi,
                                   f_trust_lo, f_trust_hi,
                                   task_min, fp_task_max,
                                   [],
                                   type_map,
                                   jdata)
    else:
        fp_tasks = _make_fp_vasp_inner_loose(modd_path, work_path,
                                   previous_path,
                                   model_devi_skip,
                                   v_trust_lo, v_trust_hi,
                                   f_trust_lo, f_trust_hi,
                                   task_min, fp_task_max,
                                   [],
                                   type_map,
                                   jdata)
    return fp_tasks

def make_fp_gaussian(iter_index,
                     jdata):
    # obtain the fp tasks we need for the ML force field refine
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0 :
        return
    # make gaussian gjf file
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    if 'user_fp_params' in jdata.keys() :
        fp_params = jdata['user_fp_params']
    else:
        fp_params = jdata['fp_params']
    cwd = os.getcwd()

    model_devi_engine = jdata.get('model_devi_engine', 'lammps')
    #infer_file = os.path.abspath('./e3_layer_infer.py')

    for ii in fp_tasks:
        os.chdir(ii)
        #os.symlink(infer_file,'e3_layer_infer.py')
        q_net = jdata['charge_net'][int(os.path.basename(ii).split('.')[1])]
        sys_data = reloadposcar('POSCAR') 
        sys_data['charge'] = q_net
        ret = make_gaussian_input(sys_data, fp_params)
        with open('input.com', 'w') as fp:
            fp.write(ret)
        os.chdir(cwd)
    _link_fp_vasp_pp(iter_index, jdata)

def make_fp (iter_index,
             jdata,
             mdata) :
    fp_style = jdata['fp_style']

    if fp_style == 'gaussian':
        make_fp_gaussian(iter_index, jdata)
    else:
        raise RuntimeError ("unsupported fp style")

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    symlink_user_forward_files(mdata=mdata, task_type="fp", work_path=work_path)

def run_fp_inner (iter_index,
                  jdata, 
                  mdata,
                  forward_files,
                  backward_files,
                  check_fin,
                  log_file = "fp.log",
                  forward_common_files=[]):
    fp_command = mdata['fp_command']
    fp_group_size = mdata['fp_group_size']
    fp_resources = mdata['fp_resources']
    mark_failure = fp_resources.get('mark_failure', False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, 'task.*'))
    fp_tasks.sort()
    if len(fp_tasks) == 0 :
        return

    fp_run_tasks = fp_tasks
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("fp" + "_user_backward_files", [])
    
    api_version = mdata.get('api_version', '0.9')
    if LooseVersion(api_version) < LooseVersion('1.0'):
        warnings.warn(f"the dpdispatcher will be updated to new version."
            f"And the interface may be changed. Please check the documents for more details")
        dispatcher = make_dispatcher(mdata['fp_machine'], mdata['fp_resources'], work_path, run_tasks, fp_group_size)
        dispatcher.run_jobs(mdata['fp_resources'],
                        [fp_command],
                        work_path,
                        run_tasks,
                        fp_group_size,
                        forward_common_files,
                        forward_files,
                        backward_files,
                        mark_failure = mark_failure,
                        outlog = log_file,
                        errlog = log_file)

    elif LooseVersion(api_version) >= LooseVersion('1.0'):
        submission = make_submission(
            mdata['fp_machine'],
            mdata['fp_resources'],
            commands=[fp_command],
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=fp_group_size,
            forward_common_files=forward_common_files,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog = log_file,
            errlog = log_file)
        submission.run_submission()

def run_fp(iter_index, 
           jdata,
           mdata) :
    fp_style = jdata['fp_style']
    fp_pp_files = jdata['fp_pp_files']
    if fp_style == "gaussian":
        forward_files = ['input.com']
        backward_files = ['output']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _gaussian_check_fin, log_file = 'output')
    else:
        raise RuntimeError ("unsupported fp style")

def _offline_model_err(iter_index,mdata,jdata):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    all_tasks = glob.glob(os.path.join(work_path, "data.*"))
    all_tasks.sort()
    run_tasks_ = all_tasks
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    type_map = jdata['type_map']
    attrs = {'pos':('node','1x1o'),'species': ('node','1x0e'), 'energy': ('graph', '1x0e'), 'forces': ('node', '1x1o')}
    cwd_ = os.getcwd(); atomic_n = ase.atom.atomic_numbers
    for dir0 in all_tasks:
        os.chdir(dir0)
        coord = np.load('set.000/coord.npy')
        type_ele = np.loadtxt('type.raw')
        species_n = [atomic_n[type_map[int(u)]] for u in type_ele]
        energy = np.load('set.000/energy.npy')
        force = np.load('set.000/force.npy')
        coord = np.array(coord, dtype=np.single)
        force = np.array(force,dtype=np.single)
        energy = np.array(energy,dtype=np.single)
        species_n = np.array(species_n,dtype=np.intc)
        lst = []
        [lst.append(dict(pos=coord[ii].reshape((len(species_n),3)),energy=energy[ii],forces=force[ii],species=np.array(species_n))) for ii in range(len(coord))]
        path = 'fp.hdf5'
        batch = Batch.from_data_list(lst, attrs); batch.dumpHDF5(path)
        os.chdir(cwd_)
    return all_tasks


def _get_err(work_path,run_tasks,trj_freq): 
    # !!!!!! we need to get the rmse and max for each config and the rmse on each batch

    cwd = os.getcwd()
    for idx, task in enumerate(run_tasks):
        
        candidate_idxs = np.loadtxt(os.path.join(work_path,os.path.basename(task).split('.')[1]+'.fp_hist_idx_cur'))
        atom_ord = np.loadtxt(os.path.join(task,'atom_ord'))
        atom_ord = [int(u) for u in atom_ord]
        if candidate_idxs.shape == ():
            candidate_idxs = [int(candidate_idxs / trj_freq)]
        else:
            candidate_idxs = [int(u/trj_freq) for u in candidate_idxs]
        os.chdir(task)
        f0 = h5py.File('f_pred0.hdf5','r')['forces'][:]; f1 = h5py.File('f_pred1.hdf5','r')['forces'][:]
        f2 = h5py.File('f_pred2.hdf5','r')['forces'][:]; f3 = h5py.File('f_pred3.hdf5','r')['forces'][:]

        f_label = h5py.File('fp.hdf5','r')['forces'][:]; n_frame = h5py.File('fp.hdf5','r')['energy'].shape[0]
        n_atoms = f_label.shape[0]; f_label = f_label.reshape((n_frame,int(n_atoms/n_frame),3))

        mol_atom_n = f_label.shape[1]; all_atom_n = f0.shape[0]
        f0 = f0.reshape((int(all_atom_n/mol_atom_n),mol_atom_n,3)); f1 = f1.reshape((int(all_atom_n/mol_atom_n),mol_atom_n,3))
        f2 = f2.reshape((int(all_atom_n/mol_atom_n),mol_atom_n,3)); f3 = f3.reshape((int(all_atom_n/mol_atom_n),mol_atom_n,3))
        f0 = f0[candidate_idxs]; f1 = f1[candidate_idxs]; f2 = f2[candidate_idxs]; f3 = f3[candidate_idxs]
        
        for ii in range(len(f0)):
            f0[ii] = f0[ii][atom_ord]; f1[ii] = f1[ii][atom_ord]
            f2[ii] = f2[ii][atom_ord]; f3[ii] = f3[ii][atom_ord]

        rmse = np.sqrt(np.mean((f0 - f_label)**2))
        rmse_single = []; max_single = []
        for ii in range(len(f0)):
            f_l = f_label[ii]; f_p = f0[ii]
            f_rmse = np.sqrt(np.mean((f_l - f_p)**2)); f_max = np.max(np.abs(f_l - f_p))
            rmse_single.append(f_rmse); max_single.append(f_max)
        fs = np.array([f0,f1,f2,f3]); fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
        model_devi_single = np.max(fs_devi,axis=-1)
        np.savez('static.npz',rmse_f=rmse,rmse_single=rmse_single,max_single=max_single,\
            model_devi_single=model_devi_single)
        os.chdir(cwd)
    return

def post_fp_gaussian (iter_index,
                      jdata,mdata):
    model_devi_jobs = jdata['model_devi_jobs']
    assert (iter_index < len(model_devi_jobs))
    model_devi_adapt_trust_lo = jdata.get('model_devi_adapt_trust_lo', False)
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    modd_path = os.path.abspath(os.path.join(iter_name, model_devi_name))
    fp_tasks = glob.glob(os.path.join(work_path, 'task.*'))
    fp_tasks.sort()
    if len(fp_tasks) == 0 :
        return

    system_index = []
    for ii in fp_tasks :
        system_index.append(os.path.basename(ii).split('.')[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()

    for ss in system_index :
        sys_output = glob.glob(os.path.join(work_path, "task.%s.*/input.log"%ss))
        sys_output.sort()
        for idx,oo in enumerate(sys_output) :
            sys = dpdata.LabeledSystem(oo, fmt = 'gaussian/log')
            if len(sys) > 0:
                sys.check_type_map(type_map = jdata['type_map'])
            if jdata.get('use_atom_pref', False):
                sys.data['atom_pref'] = np.load(os.path.join(os.path.dirname(oo), "atom_pref.npy"))
            if idx == 0:
                if jdata.get('use_clusters', False):
                    all_sys = dpdata.MultiSystems(sys, type_map = jdata['type_map'])
                else:
                    all_sys = sys
            else:
                all_sys.append(sys)
        sys_data_path = os.path.join(work_path, 'data.%s'%ss)
        # the final need to modify for nequip
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size = len(sys_output))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not os.path.isfile(os.path.join(work_path, 'data.%s'%ss, 'inference.py')):
            os.symlink(os.path.join(cwd,'inference.py'),os.path.join(work_path, 'data.%s'%ss, 'inference.py'))
            os.symlink(os.path.join(modd_path, 'task.%s'%ss+'.%06d'%(0), 'f_pred0.hdf5'), os.path.join(work_path, 'data.%s'%ss, 'f_pred0.hdf5'))
            os.symlink(os.path.join(modd_path, 'task.%s'%ss+'.%06d'%(0), 'f_pred1.hdf5'), os.path.join(work_path, 'data.%s'%ss, 'f_pred1.hdf5'))
            os.symlink(os.path.join(modd_path, 'task.%s'%ss+'.%06d'%(0), 'f_pred2.hdf5'), os.path.join(work_path, 'data.%s'%ss, 'f_pred2.hdf5'))
            os.symlink(os.path.join(modd_path, 'task.%s'%ss+'.%06d'%(0), 'f_pred3.hdf5'), os.path.join(work_path, 'data.%s'%ss, 'f_pred3.hdf5'))
        
        log_file = os.path.basename(sys_output[0]); f_read = False; atom_ord = []
        conf_file = os.path.join(sys_output[0][:-len(log_file)],'conf.dump')
        with open(conf_file,'r') as fp:
            for line in fp:
                if line.startswith('ITEM: ATOMS id type x y z fx fy fz'):
                    f_read = True
                elif f_read == True:
                    line = line.strip().split()
                    atom_ord.append(int(line[1]))
        atom_convert_ord = []
        
        for ii in range(len(jdata['type_map'])):
            for jj in range(len(atom_ord)):
                if atom_ord[jj] == ii + 1:
                    atom_convert_ord.append(jj)
        
        np.savetxt(os.path.join(work_path,'data.%s'%ss, 'atom_ord'), atom_convert_ord)

    #if model_devi_adapt_trust_lo:
    run_tasks = _offline_model_err(iter_index,mdata,jdata)
    _get_err(work_path,run_tasks,jdata['model_devi_jobs'][-1]['trj_freq'])
        

def post_fp (iter_index,
             jdata, mdata) :
    fp_style = jdata['fp_style']
    post_fp_check_fail(iter_index, jdata)
    if fp_style == 'gaussian' :
        post_fp_gaussian(iter_index, jdata, mdata)
    else :
        raise RuntimeError ("unsupported fp style")
    # clean traj
    clean_traj = True
    if 'model_devi_clean_traj' in jdata :
        clean_traj = jdata['model_devi_clean_traj']
    modd_path =  None
    if isinstance(clean_traj, bool):
        iter_name = make_iter_name(iter_index)
        if clean_traj:
            modd_path = os.path.join(iter_name, model_devi_name)
    elif isinstance(clean_traj, int):
        clean_index = iter_index - clean_traj
        if clean_index >= 0:
            modd_path = os.path.join(make_iter_name(clean_index), model_devi_name)
    if modd_path is not None:
        md_trajs = glob.glob(os.path.join(modd_path, 'task*/traj'))
        for ii in md_trajs:
            shutil.rmtree(ii)


def set_version(mdata):
    deepmd_version = '1'
    mdata['deepmd_version'] = deepmd_version
    return mdata


def _single_sys_adjusted(f_trust_lo,f_trust_hi,ii,generalML,jdata):
    # !!!!!! this function is important, it is used to adjust the trust_lo and turst_hi dynamically during the md explore 
    # !!!!!! with the combination of topo.py, we can chose the reasonable but rarely sampled configurations 

    conver_cri = jdata['fp_accurate_threshold']; max_f_cri = jdata['max_f_cri']
    rmse_f_cri = jdata['rmse_f_cri']; fp_task_max = jdata['fp_task_max']
    rmse_f_cri_hi = jdata['rmse_f_cri_hi']; max_f_cri_hi = jdata['max_f_cri_hi']
    file_path = os.path.abspath('data.'+data_system_fmt%ii+'/static.npz')
    static_infor = np.load(file_path,allow_pickle=True)
    model_max_f = static_infor['max_single']; model_rmse_f = static_infor['rmse_single']
    model_devi_f = static_infor['model_devi_single']
    static_ratio = np.loadtxt('static.'+data_system_fmt%ii)
    rmse_f_static = static_infor['rmse_f']

    this_fp_task_max = int(fp_task_max * (static_ratio[0] - conver_cri ) / (jdata['fp_accurate_soft_threshold'] - conver_cri))

    # judge whether converge
    conv = False
    if static_ratio[0] > conver_cri or this_fp_task_max == 0:
        conv = True
    elif static_ratio[-2] > jdata['model_devi_patience'] * 2 and np.max(model_max_f) < max_f_cri_hi:
        conv = True
     
    # adjust the trust_lo, trust_hi; need reload previous model_max_f and model_devi_f; maybe save as a file
    #if generalML == True:
    #    f_trust_hi = 100.
    #else:
    #    pass

    n_cand = np.where(model_rmse_f > rmse_f_cri)[0]

    adjust_hi = False
    if np.max(model_devi_f) > f_trust_hi: 
        adjust_hi = True
        n_unreason = len(np.where(model_devi_f > f_trust_hi)[0])
        f_unreason = sorted(model_max_f)[-n_unreason]
    else:
        if np.max(model_max_f) > max_f_cri * 2.:
            f_trust_hi = np.max(model_devi_f)

    if static_ratio[0] > 4*conver_cri - 3 or this_fp_task_max < 4:
        if rmse_f_static > rmse_f_cri / (min((4*conver_cri - 3), static_ratio[0])):
            f_trust_lo = f_trust_lo * ((rmse_f_cri / rmse_f_static)**0.5)
        elif np.max(model_max_f) > max_f_cri:
            f_trust_lo = f_trust_lo * max_f_cri / np.max(model_max_f)
        # then adjust if trust_lo is too low
        #n_cand = np.where(model_rmse_f > rmse_f_cri)[0]
        elif len(n_cand)/len(model_rmse_f) < 0.4:
            #n_acc = max(0,len(model_devi_f)-len(n_cand)*2)
            f_trust_lo = sorted(model_devi_f)[0]
        # adjust if the trust_hi is too high
        if adjust_hi == True:
            if rmse_f_static < rmse_f_cri:
                f_trust_hi = 1.05 * f_trust_hi
            elif rmsf_f_static > rmse_f_cri_h:
                f_trust_hi = 0.95 * f_trust_hi
            elif f_unreason > max_f_cri * 2.:
                f_trust_hi = 0.95 * f_trust_hi
    else:
        if len(n_cand)/len(model_rmse_f) < 0.4 and rmse_f_static < rmse_f_cri:
            f_acc = []
            for uu,m_devi in enumerate(model_max_f):
                if uu not in n_cand:
                    f_acc.append(m_devi)
            f_acc = np.array(f_acc)
            if len(f_acc) == 0:
                f_acc = 0.
            if np.max(f_acc) < max_f_cri:
                n_acc = int(max(0,len(model_devi_f) - len(n_cand)*2.5 - 2))
                f_trust_lo = sorted(model_devi_f)[n_acc]
        # adjust if the trust_hi is too high
        if adjust_hi == True:
            if rmse_f_static < rmse_f_cri:
                f_trust_hi = 1.05 * f_trust_hi
            elif f_unreason > max_f_cri * 2.:
                f_trust_hi = 0.95 * f_trust_hi
    return conv,f_trust_lo,f_trust_hi

def model_devi_vs_err_adjust_loose(jdata):
    # need to adjust trust_lo, trust_high automaticaaly
    # if generalML = True; need added T automatically; but nsteps and trust_lo and high using default temporaily
    # if generalML = False, need added T, nsteps automatically, primary trust_lo and high are static using previous data infor (maybe static offline by myself using analysze script) ,
    # moreover also determine the start and find sampling points acoording to the density of model_devi larger than trust_high
    
    cwd = os.getcwd()
    j_last_devi = jdata['model_devi_jobs'][-1]; trj_freq = j_last_devi["trj_freq"] 
    ensemble = j_last_devi['ensemble']; temps = j_last_devi["temps"][-1]; nsteps = j_last_devi["nsteps"]
    _idx = int(j_last_devi["_idx"]); sys_idx = j_last_devi["sys_idx"]; all_sys_idx = jdata["all_sys_idx"]
    T_list = jdata['temps_list']; f_trust_lo = jdata["model_devi_f_trust_lo"]
    patience_cut = jdata['model_devi_patience']; e3nn_md_cut = jdata['e3nn_md_patience']
    f_trust_hi = jdata["model_devi_f_trust_hi"]; generalML = jdata['generalML']
    all_sys_idx_loose = jdata["all_sys_idx_loose"]
    iter_name = make_iter_name(_idx)
    work_path = os.path.join(iter_name, fp_name)
    sys_idx_new = []; fp_accurate_threshold = jdata['fp_accurate_threshold']; fp_task_max = jdata['fp_task_max']
    fp_accurate_soft_threshold = jdata['fp_accurate_soft_threshold']
    
    for sys in sys_idx:
        static_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-1])
        large_rmse_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-2])
        # if config generate unreasonable structures five md explore and rmse is small, then assume the config itself can not converge
        unreason_num = int(np.loadtxt(os.path.join(work_path,'unreason.'+train_task_fmt % sys))[0])
        unreason_pre = int(np.loadtxt(os.path.join(work_path,'unreason.'+train_task_fmt % sys))[1])
        # obatin the history fp ratio in whole traj, if too high, do not sample candi and wait for next md 
        fp_hist_ratio = np.loadtxt(os.path.join(work_path,train_task_fmt % sys +'.fp_hist_idx_ratio'))

        if static_infor >= patience_cut or (large_rmse_infor > 2 * patience_cut):
            # if md conv or rmse is small, skip 
            pass
        elif (unreason_pre > 0 and large_rmse_infor > 2 * patience_cut) :
            # if have unreason_num, but the candi error is small
            pass
        # if a traj have unreasonable structures, and have sample candidate more than e3nn_md_cut, we don't sample this traj
        elif unreason_pre > e3nn_md_cut:
            pass
        elif fp_hist_ratio > jdata['single_traj_cand_ratio']:
            # if condi have sample enough in a traj
            pass 
        else:
            sys_idx_new.append(sys)

    all_sys_idx_loose_new = [] 
    for sys in all_sys_idx_loose:
        if os.path.isfile(os.path.join(work_path,'static.'+train_task_fmt % sys)):
            static_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-1])
            large_rmse_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-2])
            unreason_pre = int(np.loadtxt(os.path.join(work_path,'unreason.'+train_task_fmt % sys))[1])
            if static_infor >= patience_cut or (large_rmse_infor > 2 * patience_cut):
                # if a traj have unreasonable structures, md simulation must perform
                if unreason_pre > 0:
                    all_sys_idx_loose_new.append(sys)
                else:
                    pass
            else:
                all_sys_idx_loose_new.append(sys)
        else:
            all_sys_idx_loose_new.append(sys)

    os.chdir(work_path)
    next_nsteps  = max(nsteps,40000)

    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[int(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim
    
    do_md = np.loadtxt('md_cond')[0]
    part_fp = np.loadtxt('md_cond')[1]
     
    # whether do new e3nn md simulaton 
    if len(sys_idx_new) == 0:
        # if no sys_idx, do new md 
        new_simul = True
        do_md = 1
    else:
        new_simul = False
        do_md = 0
    
    if new_simul == True:
        sys_idx_new = all_sys_idx_loose_new

    jdata['all_sys_idx_loose'] = all_sys_idx_loose_new
    # whether do new md at higher temperature
    new_simul_T = True
    
    for ii in sys_idx:
        f_trust_lo_sys = _trust_limitation_check(ii, f_trust_lo)
        f_trust_hi_sys = _trust_limitation_check(ii, f_trust_hi)
        unreason_num = int(np.loadtxt('unreason.'+train_task_fmt % ii)[0])
        # !!!!!!!! if unreason and not candi, lower the trust_lo; more over if no static.npz test the part_fp
        if os.path.isfile(os.path.abspath('data.'+data_system_fmt%ii+'/static.npz')):  
            conv,f_trust_lo_sys,f_trust_hi_sys = _single_sys_adjusted(f_trust_lo_sys,f_trust_hi_sys,ii,generalML,jdata)
        else:
            unreason_pre = int(np.loadtxt('unreason.'+train_task_fmt % ii)[1])
            f_trust_lo_sys = f_trust_lo_sys; f_trust_hi_sys = f_trust_hi_sys
            accurate_ratio = np.loadtxt('static.'+train_task_fmt % ii)[0]
            this_fp_task_max = int(fp_task_max * (accurate_ratio - fp_accurate_threshold) / (fp_accurate_soft_threshold - fp_accurate_threshold))
            if this_fp_task_max <= 0:
                conv = True
            else:
                conv = False
            if unreason_pre > 0 and this_fp_task_max <= 0:
                f_trust_lo_sys = f_trust_lo_sys * 0.95
                conv = False
            

        jdata["model_devi_f_trust_lo"][ii] = f_trust_lo_sys
        if f_trust_hi_sys > 9.:
            jdata["model_devi_f_trust_hi"][ii] = f_trust_lo_sys * 3.
        else:
            jdata["model_devi_f_trust_hi"][ii] = f_trust_hi_sys
        if unreason_num >= e3nn_md_cut:
            # if md explore in a temp have unreasonable config more than e3nn_md_cut
            conv = True
        #jdata["model_devi_numb_candi_f"] = max(int(j_last_devi['nsteps']/j_last_devi['trj_freq']*jdata['fp_accurate_soft_threshold']+1), jdata['fp_task_max']*5)
        # if not converge or only part fp are calc, than we don't do md at new condition
        if conv == False:
            new_simul_T = False
    if new_simul_T == True:
        do_md = 1
    np.savetxt('md_cond',[do_md,part_fp])
    # generate next md simulation in json file 
    if new_simul_T == True:
        jdata['all_sys_idx_loose'] = all_sys_idx
        idx_temps = len(np.where(np.array(T_list)<temps)[0])
        if idx_temps == len(T_list) - 1:
            pass
        else:
            temps = T_list[idx_temps+1]; _idx += 1
    
    if _idx == len(jdata['model_devi_jobs']):
        # new simulational condition
        md_cond = {'_idx':data_system_fmt%_idx, "sys_idx":all_sys_idx, "temps":[temps], "press": [1], "nsteps": next_nsteps, "trj_freq": trj_freq, "ensemble":ensemble}
        jdata['model_devi_jobs'].append(md_cond)
    else:
        # old simulational condition but not converged
        jdata['model_devi_jobs'].append({'_idx':data_system_fmt%(_idx+1), "sys_idx":sys_idx_new,"temps":j_last_devi["temps"], "press":[1], "nsteps":j_last_devi["nsteps"], "trj_freq":j_last_devi["trj_freq"],"ensemble":j_last_devi["ensemble"]})
    
    os.chdir(cwd)
    return jdata

def model_devi_vs_err_adjust(jdata):
    # need to adjust trust_lo, trust_high automaticaaly
    # if generalML = True; need added T automatically; but nsteps and trust_lo and high using default temporaily
    # if generalML = False, need added T, nsteps automatically, primary trust_lo and high are static using previous data infor (maybe static offline by myself using analysze script) ,
    # moreover also determine the start and find sampling points acoording to the density of model_devi larger than trust_high
    cwd = os.getcwd()
    j_last_devi = jdata['model_devi_jobs'][-1]; trj_freq = j_last_devi["trj_freq"] 
    ensemble = j_last_devi['ensemble']; temps = j_last_devi["temps"][-1]; nsteps = j_last_devi["nsteps"]
    _idx = int(j_last_devi["_idx"]); sys_idx = j_last_devi["sys_idx"]; all_sys_idx = jdata["all_sys_idx"]
    T_list = jdata['temps_list']; f_trust_lo = jdata["model_devi_f_trust_lo"]
    patience_cut = jdata['model_devi_patience'] 
    f_trust_hi = jdata["model_devi_f_trust_hi"]; generalML = jdata['generalML']
    iter_name = make_iter_name(_idx)
    work_path = os.path.join(iter_name, fp_name)
    sys_idx_new = []

    for sys in sys_idx:
        static_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-1])
        large_rmse_infor = int(np.loadtxt(os.path.join(work_path,'static.'+train_task_fmt % sys))[-2])
        # if config generate unreasonable structures five md explore and rmse is small, then assume the config itself can not converge
        unreason_num = int(np.loadtxt(os.path.join(work_path,'unreason.'+train_task_fmt % sys)))
        if static_infor > patience_cut or (large_rmse_infor > 2 * patience_cut - 1 and generalML == True):
            pass
        elif (unreason_num > patience_cut and large_rmse_infor > patience_cut) or (unreason_num > patience_cut * 2 - 1):
            pass
        else:
            sys_idx_new.append(sys)

    os.chdir(work_path)
    if generalML == True:
        next_nsteps  = max(nsteps,30000)
    else:
        # !!!! need modify later
        nsteps_list = jdata['nsteps_list']
    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[int(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim
    
    do_md = np.loadtxt('md_cond')[0]
    part_fp = np.loadtxt('md_cond')[1]
    new_simul = True
    for ii in sys_idx:
        f_trust_lo_sys = _trust_limitation_check(ii, f_trust_lo)
        f_trust_hi_sys = _trust_limitation_check(ii, f_trust_hi)
        if os.path.isfile(os.path.abspath('data.'+data_system_fmt%ii+'/static.npz')):  
            conv,f_trust_lo_sys,f_trust_hi_sys = _single_sys_adjusted(f_trust_lo_sys,f_trust_hi_sys,ii,generalML,jdata)
        else:
            conv = True; f_trust_lo_sys = f_trust_lo_sys; f_trust_hi_sys = f_trust_hi_sys
        
        jdata["model_devi_f_trust_lo"][ii] = f_trust_lo_sys
        if f_trust_hi_sys > 9.:
            jdata["model_devi_f_trust_hi"][ii] = f_trust_lo_sys * 3.
        else:
            jdata["model_devi_f_trust_hi"][ii] = f_trust_hi_sys
        #jdata["model_devi_numb_candi_f"] = max(int(j_last_devi['nsteps']/j_last_devi['trj_freq']*jdata['fp_accurate_soft_threshold']+1), jdata['fp_task_max']*5)
        # if not converge or only part fp are calc, than we don't do md at new condition
        if conv == False or part_fp > 0.5:
            new_simul = False

    if len(sys_idx_new) == 0:
        new_simul = True

    if new_simul == True:
        idx_temps = len(np.where(np.array(T_list)<temps)[0])
        if generalML == True:
            if idx_temps == len(T_list) - 1:
                pass
            else:
                temps = T_list[idx_temps+1]; _idx += 1

        else:
            # !!! need modify later
            idx_nsteps = np.where(np.array(nsteps_list)<nsteps)[0]
            if idx_nsteps == len(nsteps_list) - 1:
                next_nsteps = nsteps_list[0]
                if idx_temps == len(T_list) - 1:
                    pass
                else:
                    temps = T_list[idx_temps+1]; _idx += 1
            else:
                next_nsteps = nsteps_list[idx_nsteps+1]; _idx += 1
    if _idx == len(jdata['model_devi_jobs']):
        # new simulational condition
        md_cond = {'_idx':data_system_fmt%_idx, "sys_idx":all_sys_idx, "temps":[temps], "press": [1], "nsteps": next_nsteps, "trj_freq": trj_freq, "ensemble":ensemble}
        jdata['model_devi_jobs'].append(md_cond)
    else:
        # old simulational condition but not converged
        jdata['model_devi_jobs'].append({'_idx':data_system_fmt%(_idx+1), "sys_idx":sys_idx_new,"temps":j_last_devi["temps"], "press":[1], "nsteps":j_last_devi["nsteps"], "trj_freq":j_last_devi["trj_freq"],"ensemble":j_last_devi["ensemble"]})
    os.chdir(cwd)
    return jdata 

def _json_gen(jdata,ii):
    # generate new iter in json
    from monty.serialization import dumpfn
    fparam=SHORT_CMD+'_'+str(ii)+'_param.'+jdata.get('pretty_format','json')
    dumpfn(jdata,fparam,indent=4) 
    return

def run_iter(param_file,machine_file):
    try:
        import ruamel
        from monty.serialization import loadfn
        warnings.simplefilter('ignore', ruamel.yaml.error.MantissaNoDotYAML1_1Warning)
        jdata=loadfn(param_file)
        mdata=loadfn(machine_file)
    except:
        with open (param_file, 'r') as fp :
            jdata = json.load (fp)
        with open (machine_file, 'r') as fp:
            mdata = json.load (fp)

    if mdata.get('handlers', None):
        if mdata['handlers'].get('smtp', None):
            que = queue.Queue(-1)
            queue_handler = logging.handlers.QueueHandler(que)
            smtp_handler = logging.handlers.SMTPHandler(**mdata['handlers']['smtp'])
            listener = logging.handlers.QueueListener(que, smtp_handler)
            dlog.addHandler(queue_handler)
            listener.start()
    # Convert mdata
    mdata = convert_mdata(mdata)
    max_tasks = 10000
    numb_task = 9
    record = "record.dpgen"
    iter_rec = [0, -1]
    if os.path.isfile (record) :
        with open (record) as frec :
            for line in frec :
                iter_rec = [int(x) for x in line.split()]
        dlog.info ("continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1]))

    cont = True
    ii = -1
    while cont:
        ii += 1
        iter_name=make_iter_name(ii)
        sepline(iter_name,'=')
        # !!!!! add here
        # function model_devi_vs_err_adjust used to generate new trust_lo; trust hi
        # function json_gen used to add new train iter
        # need change back later !!!!!!!!!
        if ii > iter_rec[0]:
            if jdata['generalML'] == True:
                jdata = model_devi_vs_err_adjust(jdata)
            else:
                jdata = model_devi_vs_err_adjust_loose(jdata)
            _json_gen(jdata,ii)
        for jj in range(numb_task):
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] :
                continue
            task_name="task %02d"%jj
            sepline("{} {}".format(iter_name, task_name),'-')
            ### dp train part 
            if   jj == 0 :
                log_iter ("make_train", ii, jj)
                make_train (ii, jdata, mdata)
            elif jj == 1 :
                log_iter ("run_train", ii, jj)
                run_train  (ii, jdata, mdata)
            elif jj == 2 :
                log_iter ("post_train", ii, jj)
                post_train (ii, jdata, mdata)
            ### dp explore part
                #break
            elif jj == 3 :
                log_iter ("make_model_devi",ii, jj)
                cont = make_model_devi (ii, jdata, mdata)
                if not cont :
                    break
            elif jj == 4:
                # using previous informaton to determine do md explore or using previous traj
                if ii > 0:
                    prev_name = make_iter_name(ii-1)
                    prev_work_path = os.path.join(prev_name, fp_name)
                    if os.path.isfile(os.path.join(prev_work_path, 'md_cond')):
                        do_md = np.loadtxt(os.path.join(prev_work_path,'md_cond'))[0]
                        if int(do_md) == 0:
                            do_md = False
                        else:
                            do_md = True
                    else:
                        do_md = True
                else:
                    do_md = True
                # !!!!! need modify later, now has no time to debug
                log_iter ("run_model_devi", ii, jj)
                if do_md == True:
                    run_model_devi (ii, jdata, mdata)
                else:
                    copy_model_devi (ii, jdata, mdata)
            elif jj == 5 :
                log_iter ("post_model_devi", ii, jj)
                post_model_devi (ii, jdata, mdata)
            ### dp fp part
            elif jj == 6 :
            
                log_iter ("make_fp", ii, jj)
                make_fp (ii, jdata, mdata)
            elif jj == 7 :
                log_iter ("run_fp", ii, jj)
                run_fp (ii, jdata, mdata)
            elif jj == 8 :
                log_iter ("post_fp", ii, jj)
                post_fp (ii, jdata, mdata)
            else :
                raise RuntimeError ("unknown task %d, something wrong" % jj)
            record_iter (record, ii, jj)



def gen_run(args):
    if args.PARAM and args.MACHINE:
        if args.debug:
            dlog.setLevel(logging.DEBUG)
        dlog.info ("start running")
        run_iter (args.PARAM, args.MACHINE)
        dlog.info ("finished")

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("PARAM", type=str,
                        help="The parameters of the generator")
    parser.add_argument("MACHINE", type=str,
                        help="The settings of the machine running the generator")
    args = parser.parse_args()

    logging.basicConfig (level=logging.INFO, format='%(asctime)s %(message)s')
    # logging.basicConfig (filename="compute_string.log", filemode="a", level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    logging.info ("start running")
    run_iter (args.PARAM, args.MACHINE)
    logging.info ("finished!")


if __name__ == '__main__':
    _main()
