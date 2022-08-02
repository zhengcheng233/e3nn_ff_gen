#!/usr/bin/env python
import torch
from moleculekit.molecule import Molecule
import os 
from torchmd.forcefields.forcefield import ForceField
import torch
from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.utils import LogWriter
from tqdm import tqdm
import numpy as np
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
import h5py
import time


class Parameters:
    def __init__(self, masses, mapped_atom_type, precision=torch.float, device="cpu"):
        self.masses = masses 
        self.mapped_atom_type = mapped_atom_type
        self.natoms = len(masses)
        
        self.build_parameters()
        self.precision_(precision)
        self.to_(device)

    def to_(self, device):
        self.masses = self.masses.to(device)
        self.device = device

    def precision_(self, precision):
        self.masses = self.masses.type(precision)

    def build_parameters(self):
        uqatomtypes, indexes = np.unique(self.mapped_atom_type, return_inverse=True)
        self.mapped_atom_types = torch.tensor(indexes)
        masses = torch.tensor(self.masses)
        masses.unsqueeze_(1)
        self.masses = masses

class Myclass():
    def __init__(self, config, atom_types, parameters, r_max=None):
        # information such as masses, used by the integrator
        self.par = parameters
        self.atom_types = atom_types
        self.model = build(config).to(device)
        self.n_nodes = torch.ones((1, 1), dtype=torch.long)* atom_types.shape[0]
        if r_max is None:
            self.r_max = config.r_max
        else:
            self.r_max = r_max

    def compute(self, pos, box, forces):
        data = {'pos': pos[0], 'species': self.atom_types, '_n_nodes': self.n_nodes}
        attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
        _data, _attrs = computeEdgeIndex(data, attrs, r_max=self.r_max)
        data.update(_data)
        attrs.update(_attrs)
        batch = Batch(attrs, **data).to(device)
        batch = self.model(batch)
        forces[0, :] = batch['forces'].detach()
        return [batch['energy'].item()]

#testdir = "./"
#mol = Molecule("structure.prmtop")
#mol.read("input.coor")
#mol.read("input.xsc")

precision = torch.float
device = "cuda:0"
 
mass = {1:1.00794,6:12.0107,7:14.0067,8:15.9994,16:32.065}
sym_dict = {'H':1,'C':6,'N':7,'O':8}
#ff = ForceField.create(mol, "structure.prmtop")
#parameters = Parameters(ff, mol, precision=precision, device=device)

#init_sys = h5py.File('fp.hdf5')
#mol_numAtoms = init_sys['_n_nodes'][0][0]
#mol_coords = init_sys['pos'][0:mol_numAtoms]

#mol_atypes = np.concatenate(init_sys['species'][0:mol_numAtoms])
mol_coords = []; mol_numAtoms = 0; mol_atypes = []
fr = open('sqm.pdb','r'); lines = fr.readlines(); fr.close()
for line in lines:
    line = line.strip().split()
    mol_coords.append([float(line[5]),float(line[6]),float(line[7])])
    mol_atypes.append(sym_dict[line[-1]])
mol_numAtoms = len(mol_coords)


mol_atypes = np.array(mol_atypes,dtype=np.int64)
mol_masses = [mass[u] for u in mol_atypes]
mol_masses = np.array(mol_masses)
mol_coords = np.array(mol_coords).reshape((mol_numAtoms,3,1))

#mol_masses = torch.from_numpy(mol_masses)
#mol_atypes = torch.from_numpy(mol_atypes)

parameters = Parameters(mol_masses, mol_atypes, precision=precision, device=device)

system = System(mol_numAtoms, nreplicas=1, precision=precision, device=device)
system.set_positions(mol_coords)
#box = torch.as_tensor(np.zeros((3,1)),dtype=torch.float32)
system.set_box(np.zeros((3,1)))
#system.set_box(torch.zeros(3,1).clone().detach())
system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))

config = configs.config_energy_force().model_config
config.n_dim = 32
atom_types = torch.tensor((mol_atypes))
print('************')
print(atom_types)
print(atom_types.shape)
print(parameters.mapped_atom_types)
forces = Myclass(config, atom_types, parameters)
state_dict = torch.load('./results/default/best.pt', map_location=device)
model_state_dict = {}
for key, value in state_dict.items():
    if key[:7] == 'module.':
        key = key[7:]
    model_state_dict[key] = value
forces.model.load_state_dict(model_state_dict)


langevin_temperature = 300
langevin_gamma = 0.1
timestep = 1

integrator = Integrator(system, forces, timestep, device, gamma=langevin_gamma, T=langevin_temperature)
wrapper = Wrapper(mol_numAtoms, None, device)

logger = LogWriter(path="logs/", keys=('iter','ns','epot','ekin','etot','T'), name='monitor.csv')

FS2NS = 1E-6
steps = 1000
output_period = 10
save_period = 100
traj = []

trajectroyout = "mytrajectory.npy"

time0 = time.time()
iterator = tqdm(range(1, int(steps / output_period) + 1))
Epot = forces.compute(system.pos, system.box, system.forces)
for i in iterator:
    Ekin, Epot, T = integrator.step(niter=output_period)
    wrapper.wrap(system.pos, system.box)
    currpos = system.pos.detach().cpu().numpy().copy()
    traj.append(currpos)
    if (i*output_period) % save_period == 0:
        np.save(trajectroyout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})

print(time.time()-time0)
