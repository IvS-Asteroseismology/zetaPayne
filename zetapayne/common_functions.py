import os, sys
import numpy as np
from shutil import copyfile

from zetapayne.step1_CreateGrid import run_GSSP as rungssp
from zetapayne.step1_CreateGrid import Grid as grid
from zetapayne.step1_CreateGrid import RandomGrid as randomgrid


param_names = ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
param_units = ['[K]','[cm/s^2]','[km/s]','[km/s]','[dex]']

GSSP_steps = {}
GSSP_steps['T_eff'] = [10000, 100, 250]
GSSP_steps['log(g)'] = [0.1]
GSSP_steps['v_micro'] = [0.5]
GSSP_steps['[M/H]'] = [0.1]
GSSP_steps['v*sin(i)'] = [10]

assert all([key in param_names for key in GSSP_steps])
assert all([key in GSSP_steps for key in param_names])

def parse_inp(fn='random_grid.conf'):
    opt = {}
    with open(fn) as f:
        for line in f:
            text = line[:line.find('#')]
            parts = text.split(':')
            if len(parts) < 2: continue
            name = parts[0].strip()
            arr = parts[1].split(',')
            opt[name] = [a.strip() for a in arr]
    return opt


def run_one_item(item):
    """
    Runs GSSP to create a subgrid
    """
    (run_id, subgrid, pp, pp_arr, subgrid_dir, opt) = item

    wave = [float(x) for x in opt['wavelength']]
    GSSP_run_cmd = opt['GSSP_run_cmd'][0]
    GSSP_data_path = opt['GSSP_data_path'][0]
    N_interpol_threads = int(opt['N_interpol_threads'][0])
    scratch_dir = opt['scratch_dir'][0]

    Kurucz = True
    if 'Kurucz' in opt:
        Kurucz = opt['Kurucz'][0].lower() in ['true', 'yes', '1']

    rnd_grid_dir = opt['output_dir'][0]

    inp_fn = os.path.join(subgrid_dir, 'subgrid_' + run_id + '.inp')

    # Create GSSP input file and run GSSP for the subgrid
    ok = rungssp.run_GSSP_grid(run_id, inp_fn, subgrid, wave, GSSP_run_cmd, GSSP_data_path, scratch_dir, opt['R'][0], Kurucz=Kurucz)
    if not ok:
        print('GSSP exited with error, item id '+run_id)
        return 1

    # Interpolate the subgrid
    rgs_dir = os.path.join('rgs_files', run_id)
    GRID = grid.Grid(rgs_dir)
    GRID.load()

    RND = randomgrid.RandomGrid(GRID)

    fn = run_id + '.npz'
    sp = RND.interpolate(pp_arr, N_interpol_threads)
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp, wave=wave)
    shutil.rmtree(rgs_dir, ignore_errors=True)

    print('Grid model '+run_id+' complete')

    return 0


def create_subgrid(pp, grid):
    """
    Creates a single-cell subgrid
    ---
    pp: dictionary with parameter values
    grid: dictionary with grid boundaries
    ---
    Returns disctionary with subgrid boundaries
    """
    grid_params = [p for p in param_names if grid[p][0]!=grid[p][1]]
    subgrid = {}
    for p in param_names:
        if len(GSSP_steps[p]) == 1:
            step = GSSP_steps[p][0]
        else:
            step = GSSP_steps[p][1] if pp[p] < GSSP_steps[p][0] else GSSP_steps[p][2]

        if p in grid_params:
            start = pp[p] - pp[p]%step
            subgrid[p] = [start, start + step, step]
        else:
            subgrid[p] = grid[p] + [step]

    return subgrid


def DER_SNR(flux):
    """
    Estimates SNR for a given spectrum
    ----------------------------------
    Stoehr et al, 2008. DER_SNR: A Simple & General Spectroscopic
            Signal-to-Noise Measurement Algorithm
    """
    flux = [f for f in flux if not np.isnan(f)]
    signal = np.median(flux)
    Q = []
    for i in range(2, len(flux)-2):
        q = abs(2*flux[i] - flux[i-2] - flux[i+2])
        Q.append(q)
    noise = 1.482602 * np.median(Q) / np.sqrt(6.0)
    return signal/noise


def doppler_shift(wavelength, flux, dv):
    '''
    Doppler shift a spectrum.
    -----Parameters-----
    wavelength: list
    flux: list
    dv: float
        RV shift in km/s, We use the convention where a positive dv means the object is moving away.
    -----Returns-----
    new_flux: list
        Doppler shifted flux values
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux
