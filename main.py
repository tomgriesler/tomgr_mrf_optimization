#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, MRFSequence, BLOCKS, store_optimization, create_weightingmatrix, sort_sequences, visualize_sequence

# %%
prep_modules = ['noPrep', 'TI21', 'TI300', 'T2prep40', 'T2prep80', 'T2prep120']

#%%
target_tissue = TargetTissue(661.5, 56.8, 1)

#%%
acq_block_fa = np.full(40, 15)
acq_block_tr = np.full(40, 5)
acq_block = AcquisitionBlock(acq_block_fa, acq_block_tr, 1.4)
num_acq_blocks = 12

#%% Perform optimization
sequences = optimize_sequence(target_tissue, acq_block, prep_modules, num_acq_blocks, N_iter_max=1e5)


#%%
weighting = '1/T1, 1/T2, 0'
weightingmatrix = create_weightingmatrix(target_tissue, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
prot = {
    'N': len(sequences),
    'num_acq_blocks': num_acq_blocks, 
    'prep_modules': prep_modules
}
store_optimization(sequences, prot)


#%% Compare to reference
prep_order_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']
prep_order_mod = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep80', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep80', 'TI12', 'noPrep']
prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

waittimes_jaubert = [1.2e3 - BLOCKS[name]['ti'] - BLOCKS[name]['t2te'] - sum(acq_block_tr) for name in prep_order_jaubert]
waittimes_hamilton = [1.2e3 - BLOCKS[name]['ti'] - BLOCKS[name]['t2te'] - sum(acq_block_tr) for name in prep_order_hamilton]
waittimes_mod = [1.2e3 - BLOCKS[name]['ti'] - BLOCKS[name]['t2te'] - sum(acq_block_tr) for name in prep_order_mod]


mrf_sequence_jaubert = MRFSequence(prep_order_jaubert, waittimes_jaubert, acq_block)
mrf_sequence_hamilton = MRFSequence(prep_order_hamilton, waittimes_hamilton, acq_block)
mrf_sequence_mod = MRFSequence(prep_order_mod, waittimes_mod, acq_block)

