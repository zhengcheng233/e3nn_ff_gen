#!/usr/bin/env python

from glob import glob
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata_new = loadfn('param_back0.json')
jdata_ori = loadfn('param_back1.json')

jdata_new['charge_net'] = [0] * len(jdata_ori['sys_configs'])
jdata_new['sys_configs_prefix'] = jdata_ori['sys_configs_prefix']
jdata_new['sys_configs'] = jdata_ori['sys_configs']
jdata_new['init_data_sys'] = jdata_ori['init_data_sys']
jdata_new['model_devi_f_trust_lo'] = jdata_ori['model_devi_f_trust_lo']
jdata_new['model_devi_f_trust_hi'] = jdata_ori['model_devi_f_trust_hi']
jdata_new['all_sys_idx'] = list(jdata_ori['all_sys_idx'][0:10])
model_job = jdata_ori['model_devi_jobs']
a = list(model_job[0]["sys_idx"][0:10])
jdata_new['model_devi_jobs'] = jdata_ori['model_devi_jobs']
jdata_new['model_devi_jobs'][0]['sys_idx'] = a
dumpfn(jdata_new,'param.json',indent=4)
