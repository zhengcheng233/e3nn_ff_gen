#!/bin/env python3

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

def run_VASP_Single(mdata):
    #   mdata = decide_fp_machine(mdata)
    convert_mdata(mdata, ['fp'])
    fp_command = mdata['fp_command']
    fp_group_size = mdata['fp_group_size']
    fp_resources = mdata['fp_resources']
    mark_failure = fp_resources.get('mark_failure', False)
    work_path = os.path.join("./")
    assert(os.path.isdir(work_path))

    tmp_path = 'dpgen_work' 
    dir_path = os.path.join(work_path, tmp_path)
    all_task = glob.glob(os.path.join(dir_path, ''))
    all_task.sort()

    if len(all_task) == 0 :
        return
                                                                
    fp_run_tasks = all_task
    run_tasks = all_task
    print(run_tasks)
    submission = make_submission(mdata['fp_machine'], mdata['fp_resources'], commands=[fp_command],work_path=work_path, run_tasks=run_tasks,group_size=fp_group_size, forward_common_files=[],forward_files=['INCAR', 'POTCAR', 'POSCAR', 'KPOINTS'],backward_files=['OUTCAR', 'OSZICAR', 'err', 'log', 'CONTCAR', 'XDATCAR'],outlog ='log', errlog ='err')
    submission.run_submission()

if __name__ == '__main__':
    run_VASP_Single(json.load(open('machine.json')))
