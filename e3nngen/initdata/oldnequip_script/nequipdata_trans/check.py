#!/usr/bin/env python 
import numpy as np
disindex = np.load('disindex.npz',allow_pickle=True)
index = disindex['arg']
dis = disindex['dis']
Q_data = np.load("Q_tol.npy",allow_pickle=True)
Q_l = np.load('Q_net.npy',allow_pickle=True)

for i in range(len(dis)):
    if dis[i] < 0.0001:
        q_l = Q_l[i]#Q_l[index[i]]
        q_p = Q_data[i]#[index[i]]
        #print(q_l)
        #print(q_p)
        err = q_l - q_p
        if np.abs(err) > 0.01:
            print(i)



