#!/usr/bin/env pyton
from glob import glob 
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata = loadfn('param1.json')
jdata1 = loadfn('param0.json')

model_devi_trust_lo_ori = jdata1['model_devi_f_trust_lo']
model_devi_trust_lo = list(model_devi_trust_lo_ori)
charge_net = jdata1['charge_net']

# primary training srt
init_data_sys = []
dirs = glob('./init/*/*/*/*/fp.hdf5')
for dir0 in dirs:
    init_data_sys.append(dir0[7:-8])
print(init_data_sys[0:5])
# all_sys_idx
#all_sys_idx = list(np.arange(len(jdata['sys_configs']))[270:])
all_sys_idx = list(np.arange(len(jdata1['sys_configs'])))
sys_configs = jdata1['sys_configs']

print(len(all_sys_idx))
jdata['charge_net'] = charge_net
jdata['init_data_sys'] = init_data_sys
jdata['default_training_param']['training']['systems'] = []
jdata['all_sys_idx'] = all_sys_idx
jdata['model_devi_f_trust_lo'] = model_devi_trust_lo
jdata['model_devi_jobs'] = [{'_idx':'000', 'sys_idx': all_sys_idx, 'temps':[800], 'press': [1], 'nsteps': 20000, 'trj_freq': 20, 'ensemble': 'nvt'}]
jdata['sys_configs'] = sys_configs
dumpfn(jdata,'param_test.json',indent=4)
