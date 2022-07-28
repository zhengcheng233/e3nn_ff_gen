#!/usr/bin/env python
import numpy as np

data = np.load('solvated_protein_fragments.npz',allow_pickle=True)
mono = np.load('mono_ind.npy',allow_pickle=True)

Z = data['Z'][mono]

nums = []
for i in range(len(Z)):
    z = Z[i]
    num = 0
    for j in range(len(z)):
        if z[j] > 0:
            num += 1
    nums.append(num)

ord0 = np.argsort(nums)

np.save('ord_num.npy',ord0)

