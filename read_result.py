#%%
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np

from abdominal_tools import RESULTSPATH, BLOCKS, visualize_sequence, visualize_crlb,create_weightingmatrix,sort_sequences, TargetTissue, MRFSequence

#%%
timestamp = '231115_124351'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'acq_block.pkl', 'rb') as handle: 
    acq_block = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

target_tissue = TargetTissue(prot['target_tissue']['T1'], prot['target_tissue']['T2'], prot['target_tissue']['M0'])


#%% Compare to reference
prep_order_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep']
prep_order_kvernby = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50']
prep_order_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

total_dur = 1e4

waittimes_jaubert = np.concatenate((np.full(len(prep_order_jaubert)-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + sum(acq_block.tr) for prep in prep_order_jaubert]))/(len(prep_order_jaubert)-1), [0]))

waittimes_hamilton = np.concatenate((np.full(len(prep_order_hamilton)-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + sum(acq_block.tr) for prep in prep_order_hamilton]))/(len(prep_order_hamilton)-1), [0]))

waittimes_kvernby = np.concatenate((np.full(len(prep_order_kvernby)-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + sum(acq_block.tr) for prep in prep_order_kvernby]))/(len(prep_order_kvernby)-1), [0]))

mrf_sequence_jaubert = MRFSequence(prep_order_jaubert, waittimes_jaubert)
mrf_sequence_hamilton = MRFSequence(prep_order_hamilton, waittimes_hamilton)
mrf_sequence_kvernby = MRFSequence(prep_order_kvernby, waittimes_kvernby)

mrf_sequence_jaubert.calc_crlb(acq_block, target_tissue)
mrf_sequence_hamilton.calc_crlb(acq_block, target_tissue)
mrf_sequence_hamilton.calc_crlb(acq_block, target_tissue)


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

# %%
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(16, 9))

plt.subplot(1, 3, 1)
visualize_crlb(seqs_sorted_T1, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T1, mrf_sequence_hamilton.crlb)), ls=':', label='$cost_{1, ref}$', color='tab:blue', linewidth=2)
plt.xlim(0, 1e6)
plt.ylim(0, 10)
# plt.xticks([])
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
plt.ylabel('cost function value')
plt.title('Sorted by $cost_1$')

plt.subplot(1, 3, 2)
visualize_crlb(seqs_sorted_T2, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T2, mrf_sequence_hamilton.crlb)), ls=':', label='$cost_{2, ref}$', color='tab:red', linewidth=2)
plt.xlim(0, 1e6)
plt.ylim(0, 10)
# plt.xticks([])
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
ax = plt.gca()
ax.set_yticklabels([])
plt.title('Sorted by $cost_2$')

plt.subplot(1, 3, 3)
visualize_crlb(seqs_sorted_T1T2, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T1T2, mrf_sequence_hamilton.crlb)), ls=':', label='$cost_{3, ref}$', color='tab:green', linewidth=2)
plt.xlim(0, 1e6)
plt.ylim(0, 10)
# plt.xticks([])
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
ax = plt.gca()
ax.set_yticklabels([])
plt.title('Sorted by $cost_3$')

plt.tight_layout()

# plt.savefig(f'/home/tomgr/Documents/abdominal/figures/cost_{timestamp}.png', dpi=300)

plt.show()

# %%
plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(16, 9))

plt.subplot(8, 1, 1)
visualize_sequence(seqs_sorted_T1[0], acq_block)
plt.title('Low $cost_{T1}$')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

# plt.subplot(7, 1, 2)
# visualize_sequence(seqs_sorted_T1[-1], acq_block)
# plt.title('High $cost_{T1}$')
# plt.xlim(0, 10000)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_yticks([])

plt.subplot(8, 1, 2)
visualize_sequence(seqs_sorted_T2[0], acq_block)
plt.title('Low $cost_{T2}$')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

# plt.subplot(7, 1, 4)
# visualize_sequence(seqs_sorted_T2[-1], acq_block)
# plt.title('High $cost_{T2}$')
# plt.xlim(0, 10000)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_yticks([])

plt.subplot(8, 1, 3)
visualize_sequence(seqs_sorted_T1T2[1], acq_block)
plt.title('Low $cost_{T1,T2}$')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

plt.subplot(8, 1, 4)
visualize_sequence(seqs_sorted_T1T2[500001], acq_block)
plt.title('Medium $cost_{T1,T2}$')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

plt.subplot(8, 1, 5)
visualize_sequence(seqs_sorted_T1T2[-64], acq_block)
plt.title('High $cost_{T1,T2}$')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

plt.subplot(8, 1, 6)
visualize_sequence(mrf_sequence_hamilton, acq_block)
plt.title('Hamilton')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

plt.subplot(8, 1, 7)
visualize_sequence(mrf_sequence_jaubert, acq_block)
plt.title('Jaubert')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticks([])

plt.subplot(8, 1, 8)
visualize_sequence(mrf_sequence_kvernby, acq_block)
plt.title('Kvernby')
plt.xlim(0, 10000)
ax = plt.gca()
ax.set_yticks([])
ax.set_xlabel('Time [ms]')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend([list(by_label.values())[2], list(by_label.values())[0], list(by_label.values())[1]], [list(by_label.keys())[2], list(by_label.keys())[0], list(by_label.keys())[1]], loc='upper center', ncols=3, bbox_to_anchor=(0.5, 17))

fig.subplots_adjust(hspace=1)

plt.savefig(f'/home/tomgr/Documents/abdominal/figures/sequences_{timestamp}.png', dpi=300, bbox_inches='tight')
# %%
