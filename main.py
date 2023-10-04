#%%
import numpy as np

from optimize_sequence import optimize_sequence
from abdominal_tools import TargetTissue, AcquisitionBlock, MRFSequence, BLOCKS, store_optimization, create_weightingmatrix

# %%
prep_modules = list(BLOCKS.keys())
prep_modules = ['noPrep', 'TI12', 'TI300', 'T2prep40', 'T2prep80', 'T2prep160']

#%%
target_tissue = TargetTissue(661.5, 56.8, 1)

#%%
weighting = '1, 1, 0'
weightingmatrix = create_weightingmatrix(target_tissue, weighting)

#%%
prep_order_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']
prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

acq_block_fa_jaubert = np.load('/home/tomgr/Documents/code/abdominal/fa_jaubert.npy')
acq_block_tr_jaubert = np.load('/home/tomgr/Documents/code/abdominal/tr_jaubert.npy')

waittimes_jaubert = [1.2e3 - BLOCKS[name]['ti'] - BLOCKS[name]['t2te'] - sum(acq_block_tr_jaubert) for name in prep_order_jaubert]
waittimes_hamilton = [1.2e3 - BLOCKS[name]['ti'] - BLOCKS[name]['t2te'] - sum(acq_block_tr_jaubert) for name in prep_order_hamilton]


acq_block = AcquisitionBlock(acq_block_fa_jaubert, acq_block_tr_jaubert, 1.4)

mrf_sequence_jaubert = MRFSequence(prep_order_jaubert, waittimes_jaubert, acq_block)
mrf_sequence_hamilton = MRFSequence(prep_order_hamilton, waittimes_hamilton, acq_block)

#%%
mrf_sequence_ref = mrf_sequence_jaubert

#%%
num_acq_blocks = len(mrf_sequence_ref.prep_order)

#%%
count, best_sequences, worst_sequences, crlb_array, timestamp, duration = optimize_sequence(target_tissue, acq_block, mrf_sequence_ref, prep_modules, weightingmatrix=weightingmatrix, num_acq_blocks=num_acq_blocks, track_crlbs=True, store_good=20, store_bad=20)

#%%
store_optimization(count, best_sequences, worst_sequences, crlb_array, timestamp, duration, mrf_sequence_ref, target_tissue, acq_block, reference, weightingmatrix=weightingmatrix)
# %%
