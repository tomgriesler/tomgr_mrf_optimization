#%%
import numpy as np

from abdominal_tools import MRFSequence, AcquisitionBlock, TargetTissue, BLOCKS

#%%
acq_block = AcquisitionBlock(np.full(35, 15), np.full(35, 5), 1.)

total_dur = 1e4

prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

waittimes_hamilton = np.concatenate((np.full(len(prep_order_hamilton)-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + sum(acq_block.tr) for prep in prep_order_hamilton]))/(len(prep_order_hamilton)-1), [0]))

mrf_seq = MRFSequence(prep_order_hamilton, waittimes_hamilton)

#%%
target_tissue = TargetTissue(660, 40, 1.)

mrf_seq.calc_signal(acq_block, target_tissue)
# %%
