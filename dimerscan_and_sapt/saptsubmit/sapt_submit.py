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

def run_e3nn(mdata):
    convert_mdata(mdata,['train']); all_tasks = []; commands = []
    train_command = 'molpro -n 4 -t 1 --ga-impl disk input.inp'
    mdata['train_command'] = train_command
    train_group_size = mdata['train_group_size']
    train_resources = mdata['train_resources']
    train_machine = mdata['train_machine']
    make_failure = train_resources.get('mark_failure', False)
    work_path = os.path.join("./dimer_md_for_sapt")
    all_tasks = glob(os.path.join(work_path,'near.*'))[0:4]
    all_tasks.sort()
    run_tasks = [os.path.basename(ii) for ii in all_tasks]
    commands.append(train_command)
    forward_files = ['input.inp']
    backward_files = ['input.out']
    submission = make_submission(
          train_machine,
          train_resources,
          commands=commands,
          work_path=work_path,
          run_tasks=run_tasks,
          group_size=train_group_size,
          forward_common_files=[],
          forward_files=forward_files,
          backward_files=backward_files,
          outlog = 'train.log',
          errlog = 'train.log')
    submission.run_submission()

if __name__ == '__main__':
    mdata = json.load(open('sapt_machine.json'))
    run_e3nn(mdata)
