#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, store_optimization

# %% Define possible preparation modules
prep_modules = ['noPrep', 'TI21', 'TI100', 'TI250', 'TI400', 'T2prep40', 'T2prep80', 'T2prep120']
prep_module_weights = [1, 1/4, 1/4, 1/4, 1/4, 1/3, 1/3, 1/3]
total_dur = 1e4
min_num_preps = 8

#%% Define target tissue
inversion_efficiency = 0.95
delta_B1 = 1

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
