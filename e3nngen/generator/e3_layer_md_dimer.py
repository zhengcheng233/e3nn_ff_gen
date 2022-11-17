#!/usr/bin/env python
"""
using kcal/mol for energies, K for temperatures, g/mol for masses, and ang for distance
"""
import os 
import numpy as np
import sys
import h5py
import torch
from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.utils import LogWriter
from tqdm import tqdm
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
import ase
import os
import time

langevin_temperature = int(sys.argv[1]); steps = int(sys.argv[2]); output_period = int(sys.argv[3])
bond_hi = float(sys.argv[4]); bond_lo = float(sys.argv[5]); do_md = int(sys.argv[6])
n_config = int(steps/output_period)

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
    def __init__(self, config, atom_types, parameters, center_point, indices, spring, threshold, r_max=None, device = "cuda:0"):
        # information such as masses, used by the integrator
        self.par = parameters
        self.atom_types = atom_types
        self.model = build(config).to(device)
        self.n_nodes = torch.ones((1, 1), dtype=torch.long)* atom_types.shape[0]
        self.center_point = center_point
        self.indices = indices
        self.spring = spring
        self.threshold = threshold 
        if r_max is None:
            self.r_max = config.r_max
        else:
            self.r_max = r_max

    def compute(self, pos, box, forces, device = "cuda:0"):
        data = {'pos': pos[0], 'species': self.atom_types, '_n_nodes': self.n_nodes}
        attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
        _data, _attrs = computeEdgeIndex(data, attrs, r_max=self.r_max)
        data.update(_data)
        attrs.update(_attrs)
        batch = Batch(attrs, **data).to(device)
        batch = self.model(batch)

        # add hookean potential here, the formula is like 10000*max(0,r-0.6)**2; r = \
        # sqrt((x - 1.5)^2 + (y - 1.5)^2 + (z - 1.5)^2) 

        add_forces = np.zeros((len(self.atom_types),3),dtype=np.float32)
        atom_dis = np.sqrt(np.sum((pos[self.indices] - self.center_point)**2,axis=1))
        add_indices = np.where(atom_dis > self.threshold)[0]

        if len(add_indices) > 0: 
            # unit: self.spring eV/A**2; coordinate angstrom
            for idx in add_indices:
                magnitude = self.spring * (atom_dis[idx] - self.threshold)
                assert(magnitude >= 0.)
                direction = (self.center_point - pos[self.indices[idx]]) / np.linalg.norm(self.center_point - pos[self.indices[idx]])
                add_forces[self.indices[idx]] = direction * magnitude
            
            add_forces = np.array(add_forces,dtype=np.float32) * np.array(23.0605426,dtype=np.float32)
        
        forces[0, :] = batch['forces'].detach() * np.array(23.0605426,dtype=np.float32) + add_forces
        return [batch['energy'].item()]

precision = torch.float
device = "cuda:0"

mass = {'H':1.00794,'C':12.0107,'N':14.0067,'O':15.9994,'S':32.065}
sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
lmp_map = ["C","H","N","O","S"]; flag = False
threshold = 6  
spring = 1.04  
lmp_type = []

mol_coords = []; mol_atypes = []; mol_masses = []
with open('conf.lmp','r') as fp:
    for line in fp:
        line = line.strip().split()
        if len(line) == 2 and line[-1] == 'atoms':
            mol_numAtoms = int(line[0])
        elif len(line) == 4 and line[-1] == 'xhi' and line[-2] == 'xlo':
            box_latt = max(float(line[-3]),100.)
        elif len(line) == 3 and line[0] == 'Atoms':
            flag = True
        if flag == True:
            if len(line) == 5:
                mol_coords.append([float(x) for x in line[2:5]])
                mol_atypes.append(sym_dict[lmp_map[int(line[1])-1]])
                mol_masses.append(mass[lmp_map[int(line[1])-1]])
                lmp_type.append(int(line[1]))

monomer_atoms_n = []
with open('conf.num','r') as fp:
    for line in fp:
        line = line.strip().split()
        monomer_atoms_n.append(int(line[0]))

mol_atypes = np.array(mol_atypes,dtype=np.int64)
mol_masses = np.array(mol_masses)
mol_coords = np.array(mol_coords).reshape((mol_numAtoms,3,1))

if int(do_md) == 1:
    def init_infor(mol_masses,mol_coords,mol_atypes, monomer_atoms_n):
        mol_coords = mol_coords.reshape((len(mol_coords),3))
        mass_0 = mol_masses[0:monomer_atoms_n[0]]; mass_1 = mol_masses[monomer_atoms_n[0]:]
        coord_0 = mol_coords[0:monomer_atoms_n[0]]; coord_1 = mol_coords[monomer_atoms_n[0]:]
        mol_type_0 = mol_atypes[0:monomer_atoms_n[0]]; mol_type_1 = mol_atypes[monomer_atoms_n[0]:]

        mass_cen_0 = np.sum(coord_0 * mass_0.reshape(-1,1),axis=0)/np.sum(mass_0)
        mass_cen_1 = np.sum(coord_1 * mass_1.reshape(-1,1),axis=0)/np.sum(mass_1)
        center_point = (mass_cen_0 + mass_cen_1)/2.
        idx_0 = np.argsort(np.sum((coord_0 - mass_cen_0)**2,axis=1))
        idx_1 = np.argsort(np.sum((coord_1 - mass_cen_1)**2,axis=1))
        indices = []
        for idx in idx_0:
            if mol_type_0[idx] > 1:
                indices.append(idx)
                break
        for idx in idx_1:
            if mol_type_1[idx] > 1:
                indices.append(idx + monomer_atoms_n[0])
                break
        return center_point, indices

    center_point, indices = init_infor(mol_masses,mol_coords,mol_atypes, monomer_atoms_n)

    parameters = Parameters(mol_masses, mol_atypes, precision=precision, device=device)

    system = System(mol_numAtoms, nreplicas=1, precision=precision, device=device)
    system.set_positions(mol_coords)
    system.set_box(np.zeros((3,1)))
    system.set_velocities(maxwell_boltzmann(parameters.masses, T=langevin_temperature, replicas=1))

    config = configs.config_energy_force().model_config
    #config.n_dim = 32
    atom_types = torch.tensor((mol_atypes))
    
    forces = Myclass(config, atom_types, parameters, center_point, indices, spring, threshold)
    state_dict = torch.load('../best.000.pt', map_location=device)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key[:7] == 'module.':
            key = key[7:]
        model_state_dict[key] = value
    forces.model.load_state_dict(model_state_dict)


    langevin_gamma = 0.1
    timestep = 1

    integrator = Integrator(system, forces, timestep, device, gamma=langevin_gamma, T=langevin_temperature)
    wrapper = Wrapper(mol_numAtoms, None, device)

    logger = LogWriter(path="./", keys=('iter','ns','epot','ekin','etot','T'), name='monitor.csv')

    FS2NS = 1E-6
    save_period = output_period; traj = []

    trajectroyout = "traj.npy"

    iterator = tqdm(range(1, int(steps / output_period) + 1))
    Epot = forces.compute(system.pos, system.box, system.forces)
    try:
        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=output_period)
            wrapper.wrap(system.pos, system.box)
            currpos = system.pos.detach().cpu().numpy().copy()
            traj.append(currpos)
            if (i*output_period) % save_period == 0:
                np.save(trajectroyout, np.stack(traj, axis=2))
    
            logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})
    except:
        pass

    coords = np.transpose(np.load('traj.npy')[0],(1,0,2))

def make_lammps(ii,n_step,coord,box_latt,atype):
    ptr_float_fmt = '%15.10f'; ptr_int_fmt = '%6d'; ptr_key_fmt = '%15s'

    buff = []
    buff.append('ITEM: TIMESTEP')
    buff.append('%s' %(str(ii * n_step)))
    buff.append('ITEM: NUMBER OF ATOMS')
    buff.append('%s' %(str(len(coord))))
    buff.append('ITEM: BOX BOUNDS xy xz yz ff ff ff')
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append((ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt) %(0,box_latt,0))
    buff.append('ITEM: ATOMS id type x y z fx fy fz')
    coord_fmt = ptr_int_fmt + ' ' + ptr_int_fmt + ' '+ ptr_float_fmt + ' ' + ptr_float_fmt + ' ' + ptr_float_fmt
    for ii, r0 in enumerate(coord):
        buff.append(coord_fmt % (ii+1, atype[ii], r0[0], r0[1], r0[2]))
    return '\n'.join(buff)

if int(do_md) == 1:
    if os.path.isfile('./traj/file.txt'):
        pass
    else:
        os.system('mkdir -p traj')
        with open('./traj/file.txt','w') as fp:
            fp.write('created')

    for idx,coord in enumerate(coords):
        coord_cen = np.mean(coord,axis=0)
        coord = coord - coord_cen + np.array([box_latt/2., box_latt/2., box_latt/2.])
        ret = make_lammps(idx,output_period,coord,box_latt,lmp_type)
        with open('./traj/'+str(idx*output_period)+'.lammpstrj','w') as fp:
            fp.write(ret)

    os.system('python3 topo.py %s %s %s %s' %(bond_hi, bond_lo, output_period, n_config))
    atomic_n = ase.atom.atomic_numbers

    coord = np.load('traj_deepmd/set.000/coord.npy')
    type_ele = np.loadtxt('traj_deepmd/type.raw')
    type_map = lmp_map
    species_n = [atomic_n[type_map[int(u)]] for u in type_ele]
    species_n = np.array(species_n,dtype=np.intc)
    e = np.array(0., dtype=np.single)
    lst = []
    [lst.append(dict(pos=coord[ii].reshape((len(species_n),3)),energy=e, forces = coord[ii].reshape((len(species_n),3)), species=species_n)) for ii in range(len(coord))]
    path = 'traj.hdf5'
    attrs = {'pos': ('node', '1x1o'), 'species': ('node', '1x0e'), 'energy': ('graph', '1x0e'), 'forces': ('node', '1x1o')}
    batch = Batch.from_data_list(lst, attrs)
    batch.dumpHDF5(path)
else:
    os.system('cp traj_old.hdf5 traj.hdf5')
    os.system('cp reasonable_old.txt reasonable.txt')

os.system("python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.000.pt --output_keys forces --output_path f_pred0.hdf5")
os.system("python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.001.pt --output_keys forces --output_path f_pred1.hdf5")
os.system("python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.002.pt --output_keys forces --output_path f_pred2.hdf5")
os.system("python3 inference.py --config config_energy_force --config_spec \"{'data_config.path':'traj.hdf5'}\" --model_path ../best.003.pt --output_keys forces --output_path f_pred3.hdf5")

def write_model_devi_out(devi: np.ndarray, fname: str):
    assert devi.shape[1] == 7
    header = "%10s" % "step"
    for item in 'vf':
        header += "%19s%19s%19s" % (f"max_devi_{item}", f"min_devi_{item}", f"avg_devi_{item}")
    np.savetxt(fname,
               devi,
               fmt=['%12d'] + ['%19.6e' for _ in range(6)],
               delimiter='',
               header=header)
    return devi

def calc_model_devi_f(fs: np.ndarray):
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f

def calc_model_devi(f0,f1,f2,f3,f_name,frequency,reasons):
    forces = [f0,f1,f2,f3]; forces = np.array(forces)
    devi = [np.arange(f0.shape[0]) * frequency]
    devi0 = calc_model_devi_f(forces)
    devi += devi0; devi += devi0; devi = np.vstack(devi).T
    # set unreasonable structures model_devi as very large value
    unreason = False; reasons = np.array(reasons)
    unreason_ratio = len(np.where(reasons[:,0] < 0.5)[0])/len(reasons)
    for idx,reason in enumerate(reasons):
        if reason[0] == 0 and reason[1] == 0:
            if unreason_ratio < 0.05 or (unreason_ratio < 0.1 and reasons[-1][0] > 0.5):
                devi[idx][1:] = 1000.
            else:
                unreason = True
        elif reason[0] == 0 and reason[1] == 1:
            devi[idx][1:] = 1000. 
        if unreason == True:
            # we can beleve the unreaon judge, so if nearby are reasonabel, it is reasonable
            if reason[0] == 0 or idx > len(reasons) - 3 or idx < 3:
                devi[idx][1:] = 1000.
            else:
                if reasons[idx-1][0] > 0.5:
                    pass
                else:
                    devi[idx][1:] = 1000.
    write_model_devi_out(devi,f_name)
    return

f0 = h5py.File('f_pred0.hdf5','r')['forces'][:]; f1 = h5py.File('f_pred1.hdf5','r')['forces'][:]
f2 = h5py.File('f_pred2.hdf5','r')['forces'][:]; f3 = h5py.File('f_pred3.hdf5','r')['forces'][:]
n_frame = h5py.File('traj.hdf5','r')['energy'].shape[0]; n_atoms = f0.shape[0]
f0 = f0.reshape((n_frame,int(n_atoms/n_frame),3)); f1 = f1.reshape((n_frame,int(n_atoms/n_frame),3))
f2 = f2.reshape((n_frame,int(n_atoms/n_frame),3)); f3 = f3.reshape((n_frame,int(n_atoms/n_frame),3))
reasons = []
with open('reasonable.txt','r') as fp:
    for line in fp:
        line = line.strip().split()
        reasons.append([int(line[0]),int(line[1])])
calc_model_devi(f0,f1,f2,f3,'model_devi_online.out',output_period,reasons)
if int(do_md) == 1:
    os.system('rm -rf traj')
    os.system('rm -rf traj_deepmd')
else:
    os.system('rm traj_old.hdf5')
    os.system('rm reasonable_old.txt')
time.sleep(0.5)
