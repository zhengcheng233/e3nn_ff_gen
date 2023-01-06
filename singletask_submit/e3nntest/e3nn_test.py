#!/usr/bin/env python
from glob import glob
import os

f_dirs = glob('./data.all/data*/iter.*/02.fp/data.*')
cwd = os.getcwd()
for dir0 in f_dirs[0:5]:
    #print(dir0)
    os.chdir(dir0)
    os.system('cp ' + os.path.join(cwd,'inference.py') + ' .')
    os.system("python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'fp.hdf5'}\" --model_path /root/paramtest/best.pt --output_keys forces --output_path f_pred0.hdf5")
    os.chdir(cwd)
