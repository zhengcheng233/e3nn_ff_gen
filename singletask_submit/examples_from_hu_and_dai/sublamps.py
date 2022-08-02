#!/usr/bin/env python

import os
import glob
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
#from dpgen.remote.decide_machine import decide_fp_machine
from dpgen.dispatcher.Dispatcher import Dispatcher, _split_tasks, make_dispatcher, make_submission
from dpgen import dlog

def run_LAMMPS(mdata):
    #   mdata = decide_fp_machine(mdata)
    convert_mdata(mdata, ['model_devi'])
    model_devi_group_size = mdata['model_devi_group_size']
    model_devi_resources = mdata['model_devi_resources']
    model_devi_exec = mdata['model_devi_command']

    work_path = os.path.join("./")
    assert(os.path.isdir(work_path))

    tmp_path = 'dpgen_work'
    dir_path = os.path.join(work_path, tmp_path)
    all_task = glob.glob(os.path.join(dir_path, 'task*'))
    all_task.sort()

    if len(all_task) == 0 :
        return
                                                            
    fp_run_tasks = all_task
    run_tasks = all_task
    submission = make_submission(mdata['model_devi_machine'], mdata['model_devi_resources'], commands=[model_devi_exec],work_path=work_path, run_tasks=run_tasks,group_size=model_devi_group_size, forward_common_files=[],forward_files=['conf.lmp', 'input.lammps', 'CsPbI3.pb', 'traj'],backward_files=[],outlog ='log', errlog ='err')
    submission.run_submission()

if __name__ == '__main__':
    run_LAMMPS(json.load(open('machine.json')))
