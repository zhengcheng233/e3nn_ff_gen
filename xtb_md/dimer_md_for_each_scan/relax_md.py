#!/usr/bin/env python
"""
perform md with energy and force from xtb, in this script we need to add the constrain potential 
"""

import numpy as np
from numpy.random import standard_normal
import os
from xtb.interface import Calculator
from xtb.utils import get_method
import ase
from ase import units
from ase.units import Hartree,Bohr
from ase.md.md import process_temperature
from topo_dimer import topo_bond, reasonable_judge

atomic_num = ase.data.atomic_numbers
atomic_mass = ase.data.atomic_masses
PARAMS = {'MDTemp':300,'MDdt':1.,'MDMaxStep':10000}
#PARAMS = {'MDTemp':300,'MDdt':1.,'MDMaxStep':20000}
fs = 1e-5 * np.sqrt(1.60217733e-19/1.6605402e-27)
kB = (1.380658e-23)/(1.602176462e-19)

# shift along the center of mass direction
def find_closest_distance(coord_A, coord_B):
    n_atoms1 = len(coord_A); n_atoms2 = len(coord_B)
    min_i = -1; min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = coord_A[i]
        for j in range(n_atoms2):
            r2 = coord_B[j]
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

class Dimerclass():
    def __init__(self):
        """
        center_point: the mass center 
        indices: the heavy atom idx that most near to center_point
        """
        self.spring = 0.
    
    def E_and_F(self, coord, symbol, q_net):
        # coord in bohr; energy in hartree; force in hartree/bohr
        # we convert to eV and eV/A
        # now calculate the force 
        coord = coord / Bohr
        calc = Calculator(get_method("GFN2-xTB"),symbol,coord,q_net)
        res = calc.singlepoint()
        energy = res.get_energy() * Hartree
        force = - res.get_gradient() * Hartree / Bohr
        
        # now add the constrain force 
        #force += add_forces
        return energy, force 

def KineticEnergy(v_,m_):
    return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)/len(m_)

# obtain the initial data
q_net = None; coord = []; symbol = []; mass = []
with open('input.gjf') as fp:
    for line in fp:
        line = line.strip().split()
        if len(line) == 2:
            if line[-1] == '1':
                q_net = int(line[0])
        if len(line) == 4:
            coord.append([float(line[1]),float(line[2]),float(line[3])])
            symbol.append(atomic_num[line[0]])
            mass.append(atomic_mass[atomic_num[line[0]]])

assert(q_net is not None)
assert(len(coord) > 0)

# obtain the indices from initial structure and the center
monomer_atoms_n = []
with open('dimer_num.txt','r') as fp:
    for line in fp:
        line = line.strip().split()
        monomer_atoms_n.append(int(line[0]))

coord = np.array(coord); mass = np.array(mass); symbol = np.array(symbol)
coord_0 = coord[0:monomer_atoms_n[0]]; coord_1 = coord[monomer_atoms_n[0]:]

def _maxwellboltzmanndistribution(masses, temp):
    xi = np.random.standard_normal((len(masses), 3))
    momenta = xi * np.sqrt(masses * temp)[:, np.newaxis]
    velo = momenta / masses.reshape(-1,1)
    return velo

def MaxwellBoltzmannDistribution(masses, temp):
    masses = np.concatenate(masses)
    velo = _maxwellboltzmanndistribution(masses, temp)
    return velo

# now do the md simulation 
class VelocityVerlet:
    def __init__(self,coord,symbol,q_net,snap,min_idx0,min_idx1,min_dis,threshold):
        self.maxstep = PARAMS['MDMaxStep']; self.T = PARAMS['MDTemp']*kB
        self.dt = PARAMS['MDdt']*fs; self.snap = snap
        self.KE = 0.; self.m = np.array([atomic_mass[u] for u in symbol]).reshape(-1,1)
        self.x = coord; self.symbols = symbol
        # need check whether the temperature unit is right 
        self.v = MaxwellBoltzmannDistribution(self.m, self.T)
        self.gamma = 0.01; self.threshold = threshold
        self.calc = Dimerclass()
        energy, force = self.calc.E_and_F(coord, symbol, q_net) 
        self.Epot = energy; self.N = len(self.m)
        self.v = self.Rescale(self.v)
        self.force = force
        self.q_net = q_net
        self.a = self.force / self.m 
        self.coord_save = []; self.stop = False
        self.min_idx0 = min_idx0
        self.min_idx1 = min_idx1 
        self.min_dis = min_dis
        bonds, atom_pair_lengths, proton_idx, bonds_lengths = topo_bond(len(symbol),'dimer_topo.txt','conf.bondlength')
        self.bonds = bonds; self.atom_pair_lengths = atom_pair_lengths
        self.proton_idx = proton_idx; self.bonds_lengths = bonds_lengths
        # find the minimal distance and i and j idx 
    
    def Rescale(self,v_):
        for i in range(self.N):
            Teff = 0.5*self.m[i]*np.einsum("i,i",v_[i],v_[i])/1.5
            if (Teff != 0.0):
                v_[i] *= np.sqrt(self.T/Teff)
        return v_

    def step(self,a_,x_,v_,m_,dt_,e,f,symbols,step):
        sigma = np.sqrt(2*self.T*self.gamma/m_); c1 = dt_/2. - dt_*dt_*self.gamma/8.
        c2 = dt_*self.gamma/2. - dt_*dt_*self.gamma*self.gamma/8.
        c3 = np.sqrt(dt_)*sigma/2. - dt_**1.5*self.gamma*sigma/8.
        c5 = dt_**1.5*sigma/(2 * np.sqrt(3)); c4 = self.gamma/2.*c5
        x_rand = standard_normal(size=(self.N,3)); eta_rand = standard_normal(size=(self.N,3))
        v_ += (c1*f/m_ - c2 * v_ + c3 * x_rand - c4 * eta_rand)
        old_cm = self.get_center_of_mass(x_); x = x_ + dt_ * v_ + c5 * eta_rand
        new_cm = self.get_center_of_mass(x); d = old_cm - new_cm; x = x + d
        v_ = (x - x_ - c5*eta_rand)/dt_
        e,f = self.calc.E_and_F(x, symbols, self.q_net)
        f = np.array(f)
        v_ += c1 * f/m_ - c2 * v_ + c3 * x_rand - c4 * eta_rand
        v_cm = self.get_com_velocity(v_)
        v_ -= v_cm; a = f/m_
        return x,v_,a,e,f

    def get_center_of_mass(self,x_):
        old_cm = np.dot(self.m.flatten(),x_)/self.m.sum()
        return old_cm

    def get_com_velocity(self,v_):
        return np.dot(self.m.flatten(),v_)/self.m.sum()

    def Prop(self):
        step = 0; self.KE = KineticEnergy(self.v,self.m)
        Teff = (2./3.)*self.KE/kB
        while (step < self.maxstep):
            if self.stop == True:
                step += 1
                continue
            self.x, self.v, self.a, self.Epot, self.force  = self.step(self.a, self.x, self.v, self.m, self.dt, self.Epot, self.force, self.symbols, step)
            self.KE = KineticEnergy(self.v,self.m); Teff = (2./3.)*self.KE/kB
            dis_cur = np.linalg.norm(self.x[self.min_idx0] - self.x[self.min_idx1])
            if dis_cur > self.min_dis + self.threshold or dis_cur < self.min_dis - self.threshold or dis_cur < 1.:
                self.stop = True
            if self.stop == False:
                _symbol = []; sym_dict = {1:'H',6:'C',7:'N',8:'O',16:'S'}
                for ss in self.symbols:
                    _symbol.append(sym_dict[ss])
                reason,cri_lo = reasonable_judge(self.x,_symbol,self.bonds,self.atom_pair_lengths,self.proton_idx,self.bonds_lengths) 
                if reason == False:
                    self.stop = True
            #if step % self.snap == 0:
            self.coord_save.append(self.x)
            step += 1
            
        if len(self.coord_save) < self.snap:
            if len(self.coord_save) > 1:
                _coord = [self.coord_save[-2]]
            else:
                _coord = []
        else:
            _coord = []
            for ii in range(1, len(self.coord_save)):
                if ii % self.snap == 0:
                    _coord.append(self.coord_save[ii])
        np.save('coord.npy',_coord)
        return 

# run the md simulation 
min_idx0,min_idx1,min_dis = find_closest_distance(coord_0,coord_1)
threshold = 1.5 
md = VelocityVerlet(coord, symbol, q_net, 30, min_idx0, min_idx1, min_dis, threshold)
md.Prop()
