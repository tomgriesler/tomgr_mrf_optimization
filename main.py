#%%
# from optimize_sequence import optimize_sequence
from optimize_sequence_parallel import optimize_sequence
from abdominal_tools import store_optimization, get_gitbranch, get_githash, RESULTSPATH

#%% Define optimization target
costfunction = 'crlb'
target_t1 = 660.
target_t2 = 40.
# target_t1 = [150, 828]
# target_t2 = [20, 72]
target_m0 = 1.

shots = 35
const_fa = [0., 10., 25.]
const_tr = 5.7
te = 1.
total_dur = 1e4

prep_modules = ['noPrep', 'TI21', 'TI100', 'TI250', 'TI400', 'T2prep40', 'T2prep80', 'T2prep120']
prep_module_weights = [1, 1/4, 1/4, 1/4, 1/4, 1/3, 1/3, 1/3]
min_num_preps = 8
n_iter_max = 1e6
num_junks = 1e3
num_workers = 8

inv_eff = 0.95
delta_B1 = 1.
phase_inc = 0.

#%% Perform optimization
gitbranch = get_gitbranch()
githash = get_githash()

sequences = optimize_sequence(costfunction, target_t1, target_t2, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights, min_num_preps, n_iter_max, inv_eff, delta_B1, phase_inc, num_workers=num_workers, num_junks=num_junks)

#%%
prot = {
    'costfunction': costfunction,
    'target_t1': target_t1,
    'target_t2': target_t2,
    'target_m0': target_m0,
    'shots': shots, 
    'const_fa': const_fa,
    'const_tr': const_tr,
    'te': te, 
    'total_dur': total_dur,
    'prep_modules': prep_modules, 
    'prep_module_weights': prep_module_weights,
    'min_num_preps': min_num_preps,
    'n_iter_max': n_iter_max,
    'inv_eff': inv_eff,
    'delta_B1': delta_B1,
    'phase_inc': phase_inc, 
    'gitbranch': gitbranch,
    'githash': githash
}
# store_optimization(RESULTSPATH, sequences, prot)

#%%
from pathlib import Path
altpath = Path('/media/tomgr/TOSHIBA EXT/sequences')
store_optimization(altpath, sequences, prot)
# %%
