#!/usr/bin/env python
import os 
import numpy as np
import sys
import h5py

langevin_temperature = int(sys.argv[1]); steps = int(sys.argv[2]); output_period = int(sys.argv[3])

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
        forces[0, :] = batch['forces'].detach() * 23.0605426
        return [batch['energy'].item()]


precision = torch.float
device = "cuda:0"

mass = {'H':1.00794,'C':12.0107,'N':14.0067,'O':15.9994,'S':32.065}
sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
lmp_map = ["C","H","N","O","S"]; flag = False
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

mol_atypes = np.array(mol_atypes,dtype=np.int64)
mol_masses = np.array(mol_masses)
mol_coords = np.array(mol_coords).reshape((mol_numAtoms,3,1))
parameters = Parameters(mol_masses, mol_atypes, precision=precision, device=device)

system = System(mol_numAtoms, nreplicas=1, precision=precision, device=device)
system.set_positions(mol_coords)
system.set_box(np.zeros((3,1)))
system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))

config = configs.config_energy_force().model_config
config.n_dim = 32
atom_types = torch.tensor((mol_atypes))
forces = Myclass(config, atom_types, parameters)
state_dict = torch.load('./results/default_project/default/best.pt', map_location=device)
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
for i in iterator:
    Ekin, Epot, T = integrator.step(niter=output_period)
    wrapper.wrap(system.pos, system.box)
    currpos = system.pos.detach().cpu().numpy().copy()
    traj.append(currpos)
    if (i*output_period) % save_period == 0:
        np.save(trajectroyout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})

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
