#%%
from optimize_sequence_parallel import optimize_sequence
from abdominal_tools import store_optimization, get_gitbranch, get_githash, RESULTSPATH

#%% Define optimization target
costfunction = 'crlb_orth'
optimize_positions = False
n_iter_max = 1e6
shots = 47
const_fa = [0., 10., 25.]
const_tr = 5.4
te = 1.
total_dur = 15e3

parallel = True
num_junks = 1e3
num_workers = 16

target_t1 = [1000., 1100]
target_t2 = [44., 50]
target_t1rho = [50., 90]
target_m0 = 1.

inv_eff = 0.95
delta_B1 = 1.
phase_inc = 0.

prep_modules = ['noPrep', 'TI21', 'T1rhoprep30', 'T1rhoprep50', 'T1rhoprep60', 'T2prep30', 'T2prep50', 'T2prep80']
prep_module_weights = [1/5, 1/5, 1/15, 1/15, 1/15, 2/15, 2/15, 2/15]
min_beats = 15
max_beats = 15

# prep_modules = ['noPrep', 'TI21', 'TI100', 'TI250', 'TI400', 'T2prep40', 'T2prep80', 'T2prep120']
# prep_module_weights = [1, 1/4, 1/4, 1/4, 1/4, 1/3, 1/3, 1/3]
# min_beats = 8

# prep_modules = ['noPrep', 'TI21', 'T2prep40', 'T2prep80', 'T2prep120']
# prep_module_weights = [1, 1, 1/3, 1/3, 1/3]
# min_beats = None

#%% Perform optimization
gitbranch = get_gitbranch()
githash = get_githash()

sequences = optimize_sequence(costfunction, target_t1, target_t2, target_t1rho, target_m0 , shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights, min_beats, max_beats, n_iter_max, inv_eff, delta_B1, phase_inc, optimize_positions, parallel, num_workers=num_workers, num_junks=num_junks)

#%%
prot = {
    'costfunction': costfunction,
    'optimize_positions': optimize_positions,
    'n_iter_max': n_iter_max,
    'shots': shots, 
    'const_fa': const_fa,
    'const_tr': const_tr,
    'te': te, 
    'total_dur': total_dur,
    'parallel': parallel,
    'num_junks': num_junks,
    'num_workers': num_workers,
    'target_t1': target_t1,
    'target_t2': target_t2,
    'target_t1rho': target_t1rho,
    'target_m0': target_m0,
    'inv_eff': inv_eff,
    'delta_B1': delta_B1,
    'phase_inc': phase_inc, 
    'prep_modules': prep_modules, 
    'prep_module_weights': prep_module_weights,
    'min_beats': min_beats,
    'max_beats': max_beats,
    'gitbranch': gitbranch,
    'githash': githash
}
store_optimization(RESULTSPATH, sequences, prot)

# %%
