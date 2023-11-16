#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, store_optimization

# %% Define possible preparation modules
prep_modules = ['noPrep', 'TI21', 'TI100', 'TI250', 'TI400', 'T2prep40', 'T2prep80', 'T2prep120']
prep_module_weights = [1, 1/4, 1/4, 1/4, 1/4, 1/3, 1/3, 1/3]
total_dur = 20e3
min_num_preps = 10

#%% Define target tissue
target_tissue = TargetTissue(1500, 100, 1)
inversion_efficiency = 0.95
delta_B1 = 1

#%% Create acquisition block
acq_block = AcquisitionBlock(np.full(20, 15.), np.full(20, 20.), 1.0)

#%% Perform optimization
sequences = optimize_sequence(target_tissue, acq_block, prep_modules, total_dur, prep_module_weights=prep_module_weights, min_num_preps=min_num_preps, N_iter_max=1e6, inversion_efficiency=inversion_efficiency, delta_B1=delta_B1)

#%%
prot = {
    'N': len(sequences),
    'total_dur': total_dur,
    'prep_modules': prep_modules, 
    'prep_module_weights': prep_module_weights,
    'min_num_preps': min_num_preps,
    'target_tissue': target_tissue.__dict__,
    'inversion_efficiency': inversion_efficiency,
    'delta_B1': delta_B1,
    'tr_offset': tr_offset
}
store_optimization(sequences, prot)
# %%
