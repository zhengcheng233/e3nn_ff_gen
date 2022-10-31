#!/usr/bin/env pyton
from glob import glob 
from monty.serialization import loadfn
from monty.serialization import dumpfn
import numpy as np

jdata = loadfn('param.json')
model_devi_f_trust_lo = jdata['model_devi_f_trust_lo']
jdata['model_devi_f_trust_hi'] = list(np.array(jdata['model_devi_f_trust_lo']) * 3.)
dumpfn(jdata,'param_test_1.json',indent=4)
