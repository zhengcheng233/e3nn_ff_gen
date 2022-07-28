#!/usr/bin/env python

from dpgen.remote.decide_machine import convert_mdata
from monty.serialization import loadfn 
from dpgen.generator import run

run_fp_inner = run.run_fp_inner; _gaussian_check_fin = run._gaussian_check_fin

mdata = loadfn('machine.json')
mdata = convert_mdata(mdata)
jdata = loadfn('param.json')

forward_files = ['input.com']
backward_files = ['output']

run_fp_inner(0,jdata,mdata,forward_files,backward_files,_gaussian_check_fin, log_file = 'output')
