#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, store_optimization

# %% Define possible preparation modules
prep_modules = ['noPrep', 'TI21', 'TI300', 'T2prep40', 'T2prep80', 'T2prep120']
total_dur = 20e3

#%% Define target tissue
target_tissue = TargetTissue(661.5, 56.8, 1)

#%% Create acquisition block
acq_block = AcquisitionBlock(np.full(40, 15), np.full(40, 5), 1.4)

#%% Perform optimization
sequences = optimize_sequence(target_tissue, acq_block, prep_modules, total_dur, N_iter_max=1e6)

#%%
prot = {
    'N': len(sequences),
    'total_dur': total_dur,
    'prep_modules': prep_modules, 
    'target_tissue': target_tissue.__dict__
}
store_optimization(sequences, acq_block, prot)
# %%
