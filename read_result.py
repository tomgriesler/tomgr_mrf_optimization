#%%
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np

from abdominal_tools import RESULTSPATH, BLOCKS, visualize_sequence, visualize_crlb,create_weightingmatrix,sort_sequences, TargetTissue, MRFSequence

#%%
timestamp = '231006_180622'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'acq_block.pkl', 'rb') as handle: 
    acq_block = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

target_tissue = TargetTissue(prot['target_tissue']['T1'], prot['target_tissue']['T2'], prot['target_tissue']['M0'])

#%%
weighting = '1/T1, 1/T2, 0'
weightingmatrix = create_weightingmatrix(target_tissue, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
plt.plot([len(sequence.PREP) for sequence in sequences], '.', ms=1)
plt.show()

#%% Compare to reference
prep_order_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep']
prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

waittimes_jaubert = [prot['total_dur']/12 - BLOCKS[name]['ti']-BLOCKS[name]['t2te']-sum(acq_block.tr) for name in prep_order_jaubert]
waittimes_hamilton = [prot['total_dur']/16 - BLOCKS[name]['ti']-BLOCKS[name]['t2te']-sum(acq_block.tr) for name in prep_order_hamilton]

mrf_sequence_jaubert = MRFSequence(prep_order_jaubert, waittimes_jaubert)
mrf_sequence_hamilton = MRFSequence(prep_order_hamilton, waittimes_hamilton)

mrf_sequence_jaubert.calc_crlb(acq_block, target_tissue)
mrf_sequence_hamilton.calc_crlb(acq_block, target_tissue)

#%%
visualize_crlb(sequences, weightingmatrix)
plt.axhline(np.sum(np.multiply(weightingmatrix, mrf_sequence_jaubert.crlb)), ls='--', label='Jaubert')
plt.axhline(np.sum(np.multiply(weightingmatrix, mrf_sequence_hamilton.crlb)), ls=':', label='Hamilton')
plt.ylim(0, 5)
plt.xlim(0, 400)
plt.legend()
plt.show()

#%%
visualize_sequence(sequences[0], acq_block)

# %%
