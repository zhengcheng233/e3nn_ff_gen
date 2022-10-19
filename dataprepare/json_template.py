#!/usr/bin/env python

from glob import glob
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata_net = loadfn('./json/netcharge/dpgen_105_param.json')
jdata_charge = loadfn('./json/charge/dpgen_71_param.json')

# define the param we want to renew
type_map = ["C","H","N","O","S"]
max_f_cri = 0.44; rmse_f_cri = 0.055; max_f_cri_hi = 0.52; rmse_f_cri_hi = 0.065
nsteps = 40000; trj_freq = 20; fp_accurate_threshold = 0.995
fp_accurate_soft_threshold = 0.95
model_devi_num_candi_f = int(nsteps/trj_freq*(1-fp_accurate_soft_threshold)) + 1
fp_task_max_hi = 20
# end redefine the param

# merge the element in param
charge_net0 = [0] * len(jdata_net['sys_configs'])
charge_net1 = jdata_charge['charge_net']
charge_tot = list(charge_net0) + list(charge_net1)
sys_config0 = jdata_net['sys_configs']
sys_config1 = jdata_charge['sys_configs']
sys_tot = list(sys_config0) + list(sys_config1)
_init_data = glob('./init/*/*/*/fp.hdf5')
init_data = []
for ii in _init_data:
    init_data.append(ii[0][7:])
training_reuse_iter = 2
model_trust_lo_0 = jdata_net['model_devi_f_trust_lo']
model_trust_lo_1 = jdata_charge['model_devi_f_trust_lo'][0:len(model_trust_lo_0)]
model_trust_hi_0 = jdata_net['model_devi_f_trust_hi']
model_trust_hi_1 = jdata_charge['model_devi_f_trust_hi'][0:len(model_trust_lo_1)]
model_devi_f_trust_lo = list(model_trust_lo_0) + list(model_trust_lo_1)
model_devi_f_trust_hi = []

for idx,ii in enumerate(model_trust_hi_0):
    if ii > 9.9:
        model_devi_f_trust_hi.append(model_trust_lo_0[idx]*3.)
    else:
        model_devi_f_trust_hi.append(ii)
for idx,ii in enumerate(model_trust_hi_1):
    if ii > 9.9:
        model_devi_f_trust_hi.append(model_trust_lo_1[idx]*3.)
    else:
        model_devi_f_trust_hi.append(ii)
all_sys_idx = range(len(model_trust_lo_0) + len(model_trust_lo_1))
sys_idx = all_sys_idx
# end the merge

# now renew the param and dump the json file
jdata_base = jdata_charge
jdata_base['type_map'] = type_map
jdata_base['max_f_cri'] = max_f_cri; jdata_base['rmse_f_cri'] = rmse_f_cri
jdata_base['max_f_cri_hi'] = max_f_cri_hi; jdata_base['rmse_f_cri_hi'] = rmse_f_cri_hi
jdata_base['fp_accurate_threshold'] = fp_accurate_threshold
jdata_base['fp_accurate_soft_threshold'] = fp_accurate_soft_threshold
jdata_base['model_devi_num_candi_f'] = model_devi_num_candi_f
jdata_base['fp_task_max_hi'] = fp_task_max_hi
jdata_base['charge_net'] = list(charge_tot)
jdata_base['sys_configs'] = sys_tot
jdata_base['init_data_sys'] = init_data
jdata_base['training_reuse_iter'] = training_reuse_iter
jdata_base['model_devi_trust_lo'] = list(model_devi_f_trust_lo)
jdata_base['model_devi_trust_hi'] = list(model_devi_f_trust_hi)
jdata_base['all_sys_idx'] = list(all_sys_idx)
jdata_base['model_devi_jobs'] = [{"_idx":"000","sys_idx":list(sys_idx),"temps":[100],"press":[1],"nsteps":nsteps,"trj_freq":trj_freq,"ensemble":"nvt"}]

dumpfn(jdata_base,'param.json',indent=4)


