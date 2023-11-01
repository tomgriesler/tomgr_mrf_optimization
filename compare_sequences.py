#%%
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np

from abdominal_tools import RESULTSPATH, create_weightingmatrix, sort_sequences, visualize_sequence, TargetTissue

#%%
timestamp = '231020_074613'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'acq_block.pkl', 'rb') as handle: 
    acq_block = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

target_tissue = TargetTissue(prot['target_tissue']['T1'], prot['target_tissue']['T2'], prot['target_tissue']['M0'])

#%%
seqs_sorted_T1T2 = sequences.copy()
weighting = '1/T1, 1/T2, 0'
weightingmatrix_T1T2 = create_weightingmatrix(target_tissue, weighting)
sort_sequences(seqs_sorted_T1T2, weightingmatrix_T1T2)

seqs_sorted_T1 = sequences.copy()
weighting = '1/T1, 0, 0'
weightingmatrix_T1 = create_weightingmatrix(target_tissue, weighting)
sort_sequences(seqs_sorted_T1, weightingmatrix_T1)

seqs_sorted_T2 = sequences.copy()
weighting = '0, 1/T2, 0'
weightingmatrix_T2 = create_weightingmatrix(target_tissue, weighting)
sort_sequences(seqs_sorted_T2, weightingmatrix_T2)

#%%
N = 40

plt.figure()

for i in range(N):

    plt.subplot(N, 1, i+1)

    visualize_sequence(seqs_sorted_T2[i], acq_block)
    plt.xlim(0, 10000)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticks([])
    
# %%
np.mean([len(seq.PREP) for seq in seqs_sorted_T1[:40]])
np.mean([len(seq.PREP) for seq in seqs_sorted_T2[:40]])
np.mean([len(seq.PREP) for seq in seqs_sorted_T1T2[:40]])

#%%
np.mean([seq.PREP.count(2) for seq in seqs_sorted_T1T2[:40]])

# %%
