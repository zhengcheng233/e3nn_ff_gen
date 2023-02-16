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

atomic_num = ase.data.atomic_numbers
atomic_mass = ase.data.atomic_masses
PARAMS = {'MDTemp':300,'MDdt':1.,'MDMaxStep':30000}
#PARAMS = {'MDTemp':300,'MDdt':1.,'MDMaxStep':20000}
fs = 1e-5 * np.sqrt(1.60217733e-19/1.6605402e-27)
kB = (1.380658e-23)/(1.602176462e-19)

class Dimerclass():
    def __init__(self, center_point, indices, spring, threshold):
        """
        center_point: the mass center 
        indices: the heavy atom idx that most near to center_point
        """
        self.center_point = center_point
        self.indices = indices
        self.spring = spring
        self.threshold = threshold
    
    def E_and_F(self, coord, symbol, q_net):
        # coord in bohr; energy in hartree; force in hartree/bohr
        # we convert to eV and eV/A
        
        # first calculate the additional force
        add_forces = np.zeros((len(symbol),3))
        atom_dis = np.linalg.norm((coord[self.indices] - self.center_point),axis=1)
        add_indices = np.where(atom_dis > self.threshold)[0]

        if len(add_indices) > 0:
            # unit: self.spring eV/A**2; coord angstrom
            for idx in add_indices:
                magnitude = self.spring * (atom_dis[idx] - self.threshold)
                assert(magnitude >= 0.)
                direction = (self.center_point - coord[self.indices[idx]]) / np.linalg.norm((self.center_point - coord[self.indices[idx]]))
                add_forces[self.indices[idx]] = direction * magnitude

        # now calculate the force 
        coord = coord / Bohr
        calc = Calculator(get_method("GFN2-xTB"),symbol,coord,q_net)
        res = calc.singlepoint()
        energy = res.get_energy() * Hartree
        force = - res.get_gradient() * Hartree / Bohr
        
        # now add the constrain force 
        force += add_forces
        return energy, force 

def KineticEnergy(v_,m_):
    return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)/len(m_)

# obtain the initial data
q_net = None; coord = []; symbol = []; mass = []
with open('dimer_sapt.gjf') as fp:
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
mass_0 = mass[0:monomer_atoms_n[0]]; mass_1 = mass[monomer_atoms_n[0]:]
mass_cen_0 = np.sum(coord_0 * mass_0.reshape(-1,1),axis=0)/np.sum(mass_0)
mass_cen_1 = np.sum(coord_1 * mass_1.reshape(-1,1),axis=0)/np.sum(mass_1)
center_point = (mass_cen_0 + mass_cen_1)/2.
idx_0 = np.argsort(np.sum((coord_0 - mass_cen_0)**2,axis=1)) 
idx_1 = np.argsort(np.sum((coord_1 - mass_cen_1)**2,axis=1))
indices = []
spring = 1.04; threshold = 6. 

for idx in idx_0:
    if symbol[idx] > 1:
        indices.append(idx)
        break
for idx in idx_1:
    if symbol[idx] > 1:
        indices.append(idx + monomer_atoms_n[0])
        break

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
    def __init__(self,coord,symbol,center_point,indices,spring,threshold,q_net,snap=100):
        self.maxstep = PARAMS['MDMaxStep']; self.T = PARAMS['MDTemp']*kB
        self.dt = PARAMS['MDdt']*fs; self.snap = snap
        self.KE = 0.; self.m = np.array([atomic_mass[u] for u in symbol]).reshape(-1,1)
        self.x = coord; self.symbols = symbol
        # need check whether the temperature unit is right 
        self.v = MaxwellBoltzmannDistribution(self.m, self.T)
        self.gamma = 0.01; self.spring = spring; self.threshold = threshold
        self.indices = indices; self.center_point = center_point 
        self.calc = Dimerclass(center_point, indices, spring, threshold)
        energy, force = self.calc.E_and_F(coord, symbol, q_net) 
        self.Epot = energy; self.N = len(self.m)
        self.v = self.Rescale(self.v)
        self.force = force
        self.q_net = q_net
        self.a = self.force / self.m 
        self.coord_save = []
    
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
            self.x, self.v, self.a, self.Epot, self.force  = self.step(self.a, self.x, self.v, self.m, self.dt, self.Epot, self.force, self.symbols, step)
            self.KE = KineticEnergy(self.v,self.m); Teff = (2./3.)*self.KE/kB
            if step % self.snap == 0:
                self.coord_save.append(self.x)
            step += 1
        np.save('coord.npy',self.coord_save)
        return 

# run the md simulation 
md = VelocityVerlet(coord, symbol, center_point, indices, spring, threshold, q_net, 400)
md.Prop()
