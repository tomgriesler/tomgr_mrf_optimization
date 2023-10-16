#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, store_optimization

# %% Define possible preparation modules
prep_modules = ['noPrep', 'TI21', 'TI300', 'T2prep40', 'T2prep80', 'T2prep120']
total_dur = 10e3

#%% Define target tissue
target_tissue = TargetTissue(660, 40, 1)
inversion_efficiency = 0.95
delta_B1 = 1

#%% Create acquisition block
acq_block = AcquisitionBlock(np.full(40, 15), np.full(40, 5), 1.4)

#%% Perform optimization
sequences = optimize_sequence(target_tissue, acq_block, prep_modules, total_dur, min_num_preps=10, N_iter_max=1e6, inversion_efficiency=inversion_efficiency, delta_B1=delta_B1)

#%%
prot = {
    'N': len(sequences),
    'total_dur': total_dur,
    'prep_modules': prep_modules, 
    'target_tissue': target_tissue.__dict__,
    'inversion_efficiency': inversion_efficiency,
    'delta_B1': delta_B1
}
store_optimization(sequences, acq_block, prot)
# %%
