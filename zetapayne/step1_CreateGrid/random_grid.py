# Compute synthetic spectra in a random grid
# To run do: python random_grid.py <path to configuration file>
import sys, shutil
import os, time
import numpy as np
from time import time
from shutil import copyfile

from zetapayne.common_functions import *
from CustomPool import CustomPool

###############################################################
# Read in the configuration file and create output directory
###############################################################
opt_fn = sys.argv[1]
opt = parse_inp(opt_fn)

N_models = int(opt['N_models_to_sample'][0])
N_oversample = int(opt['N_oversample'][0])
N_instances = int(opt['N_instances'][0])

rnd_grid_dir = opt['output_dir'][0]
subgrid_dir = 'subgrids'
for dn in [rnd_grid_dir, subgrid_dir]:
    if not os.path.exists(dn):
        os.makedirs(dn)

copyfile(opt_fn, os.path.join(rnd_grid_dir, '_grid.conf'))

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]


###############################################################
# Create the random grid and the GSSP subgrids
###############################################################
vsini = param_names[2]
grid_params = [p for p in param_names if grid[p][0]!=grid[p][1] and p!=vsini]

def sample_point(grid, uniform_fraction):
    pp = {}
    for p in param_names:
        if grid[p][0] == grid[p][1]: continue
        q = np.random.rand()
        if len(grid[p])==2 or (len(grid[p])==4 and q < uniform_fraction):
            v = grid[p][0] + np.random.rand() * (grid[p][1] - grid[p][0])
        else:
            v = grid[p][2] + np.random.randn() * grid[p][3]
            #if v < grid[p][2]: v = grid[p][2] + (grid[p][2] - v)
        pp[p] = v
    return pp

work = []
for i in range(N_oversample):
    pp = sample_point(grid, 0.0)
    pp_arr = []
    for j,v in enumerate(grid_params):
        pp_arr.append(pp[v])
    subgrid = create_subgrid(pp, grid)
    work_item = ('random'+str(i).zfill(6), subgrid, pp, pp_arr, subgrid_dir, opt)
    work.append(work_item)


with CustomPool(processes=N_instances) as pool:
    ret = pool.map(run_one_item, work, chunksize=1)


print('Done.')
