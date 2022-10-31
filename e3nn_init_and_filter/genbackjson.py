#!/usr/bin/env pyton
from glob import glob 
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata = loadfn('param.json')
dirs0 = glob('./poscar/charge/*/*/*single.log')
dirs1 = glob('./poscar/disulfur/*/*/*single.log')
dirs2 = glob('./poscar/netcharge/*/*/*single.log')
dirs3 = glob('./poscar/sulfur/*/*/*single.log')

#dirs_tot = dirs2 + dirs0 + dirs3 + dirs1

dirs_tot = dirs0 + dirs3 + dirs1

charge_net = []
for dir0 in dirs_tot:
    q_net = None
    with open(dir0) as fp:
        for line in fp:
            if line.startswith(" Charge = "):
                q_net = int(line.split()[2])
                charge_net.append(q_net)

all_sys_idx = list(np.arange(len(dirs_tot)))
sys_configs = []
for dir0 in dirs_tot:
    sys_configs.append([dir0[9:]])


#dumpfn(jdata,'param_back.json',indent=4)
# primary training srt
#init_data_sys = []
#dirs = glob('./init/*/*/*/fp.hdf5')
#for dir0 in dirs:
#    init_data_sys.append(dir0[7:-8])
#print(init_data_sys[0:5])

#init_data_sys = []

# all_sys_idx
#all_sys_idx = list(np.arange(len(jdata['sys_configs'])))

#jdata1 = loadfn('dpgen_55_param.json')
#model_devi_trust_lo = jdata1['model_devi_f_trust_lo']


#jdata['init_data_sys'] = init_data_sys
#jdata['default_training_param']['training']['systems'] = []
jdata['charge_net'] = charge_net
jdata['sys_configs'] = sys_configs
jdata['all_sys_idx'] = all_sys_idx
#jdata['model_devi_f_trust_lo'] = model_devi_trust_lo
jdata['model_devi_jobs'] = [{'_idx':'000', 'sys_idx': all_sys_idx, 'temps':[100], 'press': [1], 'nsteps': 40000, 'trj_freq': 20, 'ensemble': 'nvt'}]

dumpfn(jdata,'param_back.json',indent=4)
