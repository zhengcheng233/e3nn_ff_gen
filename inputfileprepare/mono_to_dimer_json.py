#!/usr/bin/env python
"""
this code is used to transfer the monomer json file to the dimer json file
"""
import os
from glob import glob 
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata0 = loadfn('param_back.json') 
sys_configs_prefix = './poscar_sapt/'

# obtain the old json infor
sys_configs_old = jdata0['sys_configs']
model_devi_f_trust_lo_old = jdata0['model_devi_f_trust_lo']
model_devi_f_trust_hi_old = jdata0['model_devi_f_trust_hi']

# obtain the new json infor
def conf_search(f_dir0,sys_confs):
    _sys_confs = []
    for sys in sys_confs:
        sys = sys[0][:-len(os.path.basename(sys[0]))]
        _sys_confs.append(sys)
    conf_idx = []
    for idx,sys in enumerate(_sys_confs):
        if sys == f_dir0:
            conf_idx.append(idx)
    assert(len(conf_idx) == 1)
    return conf_idx[0]

hdf5 = glob('./init/*/*/*/*/fp.hdf5')
_hdf5 = []
for file0 in hdf5:
    f_name = os.path.basename(file0); f_name = file0[:-len(f_name)-1]
    _hdf5.append(f_name[7:])
confs = glob('./poscar_sapt/*/*/*/dimer_sapt.log')
_confs = [] 
for file0 in confs:
    f_name = file0[14:]
    _confs.append([f_name])
charges = []
for file0 in confs:
    f_name = os.path.basename(file0); q_net = None
    f_name = file0[:-len(f_name)] + 'dimer_sapt.gjf'
    with open(f_name,'r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 2 and line[-1] == '1':
                q_net = int(line[0])
    charges.append(q_net)
trust_lo = []; trust_hi = []
# trust_lo and trust_hi is min(trust_mon_A, trust_mon_B), later for A-B dimer, we need to record the monomer composition
for idx,conf in enumerate(_confs):
    f_dir0 = conf[0][:-len(os.path.basename(conf[0]))]
    conf_idx = conf_search(f_dir0, sys_configs_old) 
    min_lo = 0.9 * min(model_devi_f_trust_lo_old[conf_idx],model_devi_f_trust_lo_old[conf_idx])
    min_hi = 0.9 * min(model_devi_f_trust_hi_old[conf_idx],model_devi_f_trust_hi_old[conf_idx])
    if min_hi < 1.3 * min_lo:
        min_hi = 1.5 * min_lo
    trust_lo.append(min_lo); trust_hi.append(min_hi)

jdata0['sys_configs'] = _confs
jdata0['init_data_sys'] = _hdf5
jdata0['charge_net'] = charges
jdata0['model_devi_f_trust_lo'] = trust_lo
jdata0['model_devi_f_trust_hi'] = trust_hi
jdata0['all_sys_idx'] = list(np.arange(len(charges)))
jdata0['model_devi_jobs'] = [{"_idx":"000","sys_idx":list(np.arange(len(charges))),"temps":[100],"press":[1],"nsteps":40000,"trj_freq":20,"ensemble":"nvt"}]
jdata0["all_sys_idx_loose"] = list(np.arange(len(charges)))

dumpfn(jdata0,'param_1.json',indent=4)
