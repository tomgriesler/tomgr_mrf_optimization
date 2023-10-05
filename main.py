#%%
import numpy as np
import matplotlib.pyplot as plt

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, MRFSequence, BLOCKS, store_optimization, create_weightingmatrix, sort_sequences, visualize_sequence,  visualize_crlb

# %% Define possible preparation modules
prep_modules = ['noPrep', 'TI21', 'TI300', 'T2prep40', 'T2prep80', 'T2prep120']
total_dur = 10e3

#%% Define target tissue
target_tissue = TargetTissue(661.5, 56.8, 1)

#%% Create acquisition block
acq_block_fa = np.full(40, 15)
acq_block_tr = np.full(40, 5)
acq_block = AcquisitionBlock(acq_block_fa, acq_block_tr, 1.4)

#%% Perform optimization
sequences = optimize_sequence(target_tissue, acq_block, prep_modules, total_dur, N_iter_max=1e5)

#%%
weighting = '1/T1, 1/T2, 0'
weightingmatrix = create_weightingmatrix(target_tissue, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
plt.plot([len(sequence.prep_order) for sequence in sequences], '.', ms=1)

#%% Compare to reference
prep_order_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']
prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

waittimes_jaubert = [total_dur/12 - BLOCKS[name]['ti']-BLOCKS[name]['t2te']-sum(acq_block_tr) for name in prep_order_jaubert]
waittimes_hamilton = [total_dur/16 - BLOCKS[name]['ti']-BLOCKS[name]['t2te']-sum(acq_block_tr) for name in prep_order_hamilton]

mrf_sequence_jaubert = MRFSequence(prep_order_jaubert, waittimes_jaubert, acq_block)
mrf_sequence_hamilton = MRFSequence(prep_order_hamilton, waittimes_hamilton, acq_block)

mrf_sequence_jaubert.calc_crlb(target_tissue)
mrf_sequence_hamilton.calc_crlb(target_tissue)

#%%
visualize_crlb(sequences, weightingmatrix)
plt.axhline(np.trace(weightingmatrix@np.sqrt(np.abs(mrf_sequence_jaubert.crlb))), ls='--', label='Jaubert')
plt.axhline(np.trace(weightingmatrix@np.sqrt(np.abs(mrf_sequence_hamilton.crlb))), ls=':', label='Hamilton')
plt.ylim(0, 20)
plt.legend()

#%%
prot = {
    'N': len(sequences),
    'prep_modules': prep_modules, 
    'target_tissue': target_tissue.__dict__
}
store_optimization(sequences, prot)




# %%
np.trace(weighting@np.sqrt(np.abs(mrf_sequence_jaubert.crlb)))