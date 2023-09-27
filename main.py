#%%
from optimize_sequence import optimize_sequence, store_optimization, return_reference
from abdominal_tools import TargetTissue, BLOCKS

# %%
reference = 'jaubert'
reference = 'hamilton'

prep_modules = list(BLOCKS.keys())
prep_modules = ['noPrep', 'TI12', 'TI300', 'T2prep40', 'T2prep80', 'T2prep160']

#%%
target_tissue = TargetTissue(661.5, 56.8, 1)

acquisition_block, mrf_sequence_ref = return_reference(reference)
mrf_sequence_ref.calc_crlb(target_tissue, acquisition_block.TE)
num_acq_blocks = len(mrf_sequence_ref.prep_order)

#%%
count, best_sequences, worst_sequences, crlb_array, timestamp, duration = optimize_sequence(target_tissue, acquisition_block, mrf_sequence_ref, prep_modules, num_acq_blocks, track_crlbs=True, store_good=20, store_bad=20)

#%%
store_optimization(count, best_sequences, worst_sequences, crlb_array, timestamp, duration, mrf_sequence_ref, target_tissue, acquisition_block, reference)
# %%
