#!/usr/bin/env python
"""
perform md with energy and force from xtb, in this script we need to add the constrain potential 
"""

import numpy as np
from numpy.random import standard_normal
import os
import gaps
import ebf
from xtb.inferface import Calculator
from xtb.utils import get_method
import ase

atomic_num = ase.data.atomic_numbers
PARAMS = {'MDTemp':500,'MDdt':1.,'MNHChain':2,'MDMaxStep':1000000,'MDV0':'Thermal','MDThermostat':'Langevin'}
IDEALGASR = 8.3144621 # J/molK
KJPERHARTREE = 2625.499638
fs = 1e-5 * np.sqrt(1.60217733e-19/1.6605402e-27)
sym_dict = {'H':1,'C':6,'N':7,'O':8}
JOULEPERHARTREE = KJPERHARTREE*1000.0
ATOMICMASSES = {1:1.008,6:12.011,7:14.007,8:15.999}
kB = (1.380658e-23)/(1.602176462e-19)

def KineticEnergy(v_,m_):
    return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)/len(m_)

class VelocityVerlet:
    def __init__(self,inputs,insyms,force,velocs,energy):
        #self.alpha = alpha#; self.findexs = findexs
        self.maxstep = PARAMS['MDMaxStep']; self.T = PARAMS['MDTemp']*kB
        self.dt = PARAMS['MDdt']*fs; self.Epot = energy
        self.KE = 0.; self.high = 0; self.step10 = 50
        self.m = (np.array([ATOMICMASSES[sym_dict[x]] for x in insyms])).reshape(-1,1)
        self.x = inputs; self.v = velocs; self.force = force
        self.a = self.force/self.m; self.symbols = insyms
        self.primstr0 = 'soap cutoff=3 l_max=3 n_max=6 atom_sigma=0.4 n_Z=4 Z={1 6 7 8} n_species=4 species_Z={1 6 7 8}'
        self.primzeta0 = 2.5
        self.N = len(self.m); self.gamma = 0.02
        self.n_atom = []; self.indexs = []; self.indexs_hypair = []
        self.indexs_hy = []; self.do_frag = False; self.orindexs = []
        self.gpprim = gaps.single_GP(self.primstr0,self.primzeta0)

    def Rescale(self,v_):
        for i in range(self.N):
            Teff = 0.5*self.m[i]*np.einsum("i,i",v_[i],v_[i])/1.5
            if (Teff != 0.0):
                v_[i] *= np.sqrt(self.T/Teff)
        return v_

    def EnergyAndForce(self,coord0,symbol0,step):
        """only use the energy and force from center"""
        frc_coef = False
        if step % 30 == 0 or self.do_frag == True:
            os.system('cp frame_0.frg frame_'+str(step)+'.frg')
            fr = open('frame_'+str(step)+'.keys','w'); fr.writelines('#!/usr/bin/env bash'+'\n')
            fr.writelines('#'+'\n'); fr.writelines('# main'+'\n')
            fr.writelines(''' main_chk='frame_'''+str(step)+'''.chk' '''+'\n')
            fr.writelines(''' main_mem='4gb' '''+'\n'); fr.writelines(' main_nproc=3'+'\n')
            fr.writelines(''' main_method='wb97xd' '''+'\n'); fr.writelines(''' main_basis='6-31g' '''+'\n')
            fr.writelines('#end'+'\n'); fr.writelines('\n'); fr.writelines('#lsqc'+'\n')
            fr.writelines(' lsqc_dis=4'+'\n'); fr.writelines(' lsqc_maxsubfrag=4'+'\n')
            fr.writelines(''' lsqc_frag='read' '''+'\n'); fr.writelines('#end'+'\n')
            fr.writelines('\n'); fr.writelines('# main'+'\n')
            fr.writelines(' main_charge=0'+'\n'); fr.writelines(' main_multiplicity=1'+'\n')
            fr.writelines('#end'+'\n'); fr.writelines('\n'); fr.close()
            indexs,indexs_hypair,centerindexs = ebf.fragold(coord0,symbol0,step)
            self.indexs = indexs; self.indexs_hypair = indexs_hypair; self.centerindexs = centerindexs
            self.n_atom = [(len(self.indexs[i]) + len(self.indexs_hypair[i])) for i in range(len(self.indexs))]
            self.orindexs = []
            for i in range(len(centerindexs)):
                orindex = np.array(self.indexs[i])[self.centerindexs[i][0:len(self.indexs[i])]]
                self.orindexs.append(orindex)

        coord_sub0 = []; symbol_sub0 = []
        for i in range(len(self.indexs)):
            add_hy = self.indexs_hypair[i]; index = self.indexs[i]
            c0 = [coord0[ind] for ind in index]
            s0 = [symbol0[ind] for ind in index]
            for j in range(len(add_hy)):
                h_c = ebf.Hydrogen(coord0,add_hy[j])
                c0.append(h_c); s0.append('H')
            coord_sub0.append(c0); symbol_sub0.append(s0)
        if step % 30 == 0 or self.do_frag == True:
            ebf.xyzwrite(step,coord0,symbol0)
        E,F,large_fricprim,Tole,do_frag = self.gpprim.predict(step,coord_sub0,symbol_sub0,\
                coord0,symbol0,self.n_atom,self.centerindexs,self.indexs,self.orindexs,self.do_frag)
        self.do_frag = do_frag
        if large_fricprim:
            frc_coef = True
        return E,F,frc_coef

    def step(self,a_,x_,v_,m_,dt_,e,f,symbols,step):
        if self.step10 < 50:
            self.gamma = 0.03
        else:
            self.gamma = 0.01
        sigma = np.sqrt(2*self.T*self.gamma/m_); c1 = dt_/2. - dt_*dt_*self.gamma/8.
        c2 = dt_*self.gamma/2. - dt_*dt_*self.gamma*self.gamma/8.
        c3 = np.sqrt(dt_)*sigma/2. - dt_**1.5*self.gamma*sigma/8.
        c5 = dt_**1.5*sigma/(2 * np.sqrt(3)); c4 = self.gamma/2.*c5
        x_rand = standard_normal(size=(self.N,3)); eta_rand = standard_normal(size=(self.N,3))
        v_ += (c1*f/m_ - c2 * v_ + c3 * x_rand - c4 * eta_rand)
        old_cm = self.get_center_of_mass(x_); x = x_ + dt_ * v_ + c5 * eta_rand
        new_cm = self.get_center_of_mass(x); d = old_cm - new_cm; x = x + d
        v_ = (x - x_ - c5*eta_rand)/dt_
        e,f,frc_coef = self.EnergyAndForce(x,symbols,step)
        if frc_coef == True:
            self.step10 = 0
        f = np.array(f)*27.2113966
        v_ += c1 * f/m_ - c2 * v_ + c3 * x_rand - c4 * eta_rand
        v_cm = self.get_com_velocity(v_)
        v_ -= v_cm; a = f/m_; self.step10 += 1
        return x,v_,a,e,f

    def get_center_of_mass(self,x_):
        old_cm = np.dot(self.m.flatten(),x_)/self.m.sum()
        return old_cm

    def get_com_velocity(self,v_):
        return np.dot(self.m.flatten(),v_)/self.m.sum()

    def Prop(self):
        step = 0; fr = open('traj6.txt','w')
        self.KE = KineticEnergy(self.v,self.m)
        Teff = (2./3.)*self.KE/kB
        while (step < self.maxstep):
            self.x, self.v, self.a, self.Epot, self.force  = self.step(self.a, self.x, self.v, self.m, self.dt, self.Epot, self.force, self.symbols, step)
            self.KE = KineticEnergy(self.v,self.m); Teff = (2./3.)*self.KE/kB
            if Teff > 600:
                self.step10 = 0
            if (step%10 == 0):
                fr.writelines('step %s %s %s' %(str(step),str(self.Epot),str(Teff))+'\n')
                for i in range(len(self.symbols)):
                    fr.writelines('%s %s %s %s %s %s %s %s %s %s' %(self.symbols[i],\
                        str(self.x[i][0]),str(self.x[i][1]),str(self.x[i][2]),\
                        str(self.force[i][0]),str(self.force[i][1]),str(self.force[i][2]),\
                        str(self.v[i][0]),str(self.v[i][1]),str(self.v[i][2]))+'\n')
            step += 1
        fr.close()
        return

if __name__ == '__main__':
    q_net = None; coord = []; symbol = []
    with open('conf.000_000/dimer_sapt.gjf') as fp:
        for line in fp：
            line = line.strip().split()
            if len(line) == 2:
                if line[-1] == '1':
                    q_net = int(line[0])
            if len(line) == 4:
                coord.append([float(line[1]),float(line[2]),float(line[3])])
                symbol.append(atomic_num[int(line[0])])
    assert(q_net is not)
    assert(len(coord) > 0)
    calc = Calculator(get_method("GFN2-xTB"),symbol,coord,q_net)
    res = calc.singlepoint()
    energy = res.get_energy()
    # pay attention !!!! check the difference of force and gradient
    force = - res.get_gradient()
    

    # then run constrained md
    md = VelocityVerlet(coords,symbols,forces,velocs,energy)
    md.Prop()

