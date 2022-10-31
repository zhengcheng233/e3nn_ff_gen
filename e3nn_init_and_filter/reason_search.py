#!/usr/bin/env python
from glob import glob
import numpy as np

dirs = glob('./task.*')
for dir0 in dirs:
    reason = np.loadtxt(dir0+'/reasonable.txt')
    if np.sum(reason) < len(reason):
        print(dir0)
