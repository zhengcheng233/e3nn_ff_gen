#!/usr/bin/env python
"""
static the failed config during the md exploration 
"""
import numpy as np
from monty.serialization import loadfn
import sys
from glob import glob
import os

json_f = loadfn('dpgen_9_param.json')
all_idx = json_f['all_sys_idx']
sys_configs = json_f['sys_configs']

iter_name = '%06d'%(int(sys.argv[1]))
f_names = glob('./iter.'+iter_name+'/01.model_devi/task.*')

def check(reason):
    stable = True
    for rea in reason:
        if rea < 0.5:
            stable = False
    return stable

for f_name in f_names:
    f_reason = os.path.join(f_name,'reasonable.txt')
    reason = np.loadtxt(f_reason,dtype=int)
    stable = check(reason)
    if stable == False:
        fail_idx = int(os.path.basename(f_name).split('.')[-2])
        fail_config = sys_configs[fail_idx]
        print(fail_idx)
        print(fail_config)        
