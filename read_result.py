#%%
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np

from abdominal_tools import RESULTSPATH, BLOCKS, visualize_sequence, visualize_cost,create_weightingmatrix,sort_sequences, MRFSequence

#%%
timestamp = '231206_084151'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

#%%
for sequence in sequences:
    sequence.decompress()

#%%
costfunction = prot['costfunction']
target_t1 = prot['target_t1']
target_t2 = prot['target_t2']
target_m0 = prot['target_m0']
shots = prot['shots']
const_fa = prot['const_fa']
const_tr = prot['const_tr']
te = prot['te']
total_dur = prot['total_dur']
phase_inc = prot['phase_inc']
inv_eff = prot['inv_eff']
delta_B1 = prot['delta_B1']

#%% Compare to reference
beats_jaubert = 24
beats_kvernby = 24
beats_hamilton = 24

prep_blocks_jaubert = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep']
prep_order_jaubert = np.concatenate((np.tile(prep_blocks_jaubert, reps=beats_jaubert//len(prep_blocks_jaubert)), prep_blocks_jaubert[:beats_jaubert%len(prep_blocks_jaubert)]))
waittimes_jaubert = np.full(beats_jaubert-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + shots*const_tr for prep in prep_order_jaubert]))/(beats_jaubert-1)
prep_jaubert = [BLOCKS[name]['prep'] for name in prep_order_jaubert]
ti_jaubert = [BLOCKS[name]['ti'] for name in prep_order_jaubert]
t2te_jaubert = [BLOCKS[name]['t2te'] for name in prep_order_jaubert]
fa_jaubert = np.full(beats_jaubert * shots, 15.)
tr_jaubert = np.full(beats_jaubert*shots, 0)
for ii in range(len(waittimes_jaubert)):
    tr_jaubert[(ii+1)*shots-1] += waittimes_jaubert[ii]*1e3
ph_jaubert = phase_inc*np.arange(beats_jaubert*shots).cumsum()
mrf_sequence_jaubert = MRFSequence(beats_jaubert, shots, fa_jaubert, tr_jaubert, ph_jaubert, prep_jaubert, ti_jaubert, t2te_jaubert, const_tr, te)

prep_blocks_kvernby = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50']
prep_order_kvernby = np.concatenate((np.tile(prep_blocks_kvernby, reps=beats_kvernby//len(prep_blocks_kvernby)), prep_blocks_kvernby[:beats_kvernby%len(prep_blocks_kvernby)]))
waittimes_kvernby = np.full(beats_kvernby-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + shots*const_tr for prep in prep_order_kvernby]))/(beats_kvernby-1)
prep_kvernby = [BLOCKS[name]['prep'] for name in prep_order_kvernby]
ti_kvernby = [BLOCKS[name]['ti'] for name in prep_order_kvernby]
t2te_kvernby = [BLOCKS[name]['t2te'] for name in prep_order_kvernby]
fa_kvernby = np.full(beats_kvernby * shots, 15.)
tr_kvernby = np.full(beats_kvernby*shots, 0)
for ii in range(len(waittimes_kvernby)):
    tr_kvernby[(ii+1)*shots-1] += waittimes_kvernby[ii]*1e3
ph_kvernby = phase_inc*np.arange(beats_kvernby*shots).cumsum()
mrf_sequence_kvernby = MRFSequence(beats_kvernby, shots, fa_kvernby, tr_kvernby, ph_kvernby, prep_kvernby, ti_kvernby, t2te_kvernby, const_tr, te)

prep_blocks_hamilton = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']
prep_order_hamilton = np.concatenate((np.tile(prep_blocks_hamilton, reps=beats_hamilton//len(prep_blocks_hamilton)), prep_blocks_hamilton[:beats_hamilton%len(prep_blocks_hamilton)]))
waittimes_hamilton = np.full(beats_hamilton-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + shots*const_tr for prep in prep_order_hamilton]))/(beats_hamilton-1)
prep_hamilton = [BLOCKS[name]['prep'] for name in prep_order_hamilton]
ti_hamilton = [BLOCKS[name]['ti'] for name in prep_order_hamilton]
t2te_hamilton = [BLOCKS[name]['t2te'] for name in prep_order_hamilton]
fa_hamilton = np.full(beats_hamilton * shots, 15.)
tr_hamilton = np.full(beats_hamilton*shots, 0)
for ii in range(len(waittimes_hamilton)):
    tr_hamilton[(ii+1)*shots-1] += waittimes_hamilton[ii]*1e3
ph_hamilton = phase_inc*np.arange(beats_hamilton*shots).cumsum()
mrf_sequence_hamilton = MRFSequence(beats_hamilton, shots, fa_hamilton, tr_hamilton, ph_hamilton, prep_hamilton, ti_hamilton, t2te_hamilton, const_tr, te)

#%%
mrf_sequence_jaubert.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)
mrf_sequence_kvernby.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)
mrf_sequence_hamilton.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)

#%%
seqs_sorted_T1T2 = sequences.copy()
weighting = '1/T1, 1/T2, 0'
weightingmatrix_T1T2 = create_weightingmatrix(target_t1, target_t2, target_m0, weighting)
sort_sequences(seqs_sorted_T1T2, weightingmatrix_T1T2)

seqs_sorted_T1 = sequences.copy()
weighting = '1/T1, 0, 0'
weightingmatrix_T1 = create_weightingmatrix(target_t1, target_t2, target_m0, weighting)
sort_sequences(seqs_sorted_T1, weightingmatrix_T1)

seqs_sorted_T2 = sequences.copy()
weighting = '0, 1/T2, 0'
weightingmatrix_T2 = create_weightingmatrix(target_t1, target_t2, target_m0, weighting)
sort_sequences(seqs_sorted_T2, weightingmatrix_T2)

# %%
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(16, 9))

plt.subplot(1, 3, 1)
visualize_cost(seqs_sorted_T1, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T1, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{1, ref}$', color='tab:blue', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
plt.ylabel('cost function value')
plt.title('Sorted by $cost_1$')

plt.subplot(1, 3, 2)
visualize_cost(seqs_sorted_T2, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T2, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{2, ref}$', color='tab:red', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
ax = plt.gca()
ax.set_yticklabels([])
plt.title('Sorted by $cost_2$')

plt.subplot(1, 3, 3)
visualize_cost(seqs_sorted_T1T2, weightingmatrix_T1T2)
plt.axhline(np.sum(np.multiply(weightingmatrix_T1T2, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{3, ref}$', color='tab:green', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
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

n_subplots = 8
ii = 1

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_T1[0], True)
plt.title('Low $cost_{T1}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

# plt.subplot(n_subplots, 1, ii)
# visualize_sequence(seqs_sorted_T1[-1])
# plt.title('High $cost_{T1}$')
# plt.xlim(0, total_dur)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_ylim(0, 1.1*np.max(const_fa))
# ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_T2[0], True)
plt.title('Low $cost_{T2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

# plt.subplot(n_subplots, 1, ii)
# visualize_sequence(seqs_sorted_T2[-1])
# plt.title('High $cost_{T2}$')
# plt.xlim(0, total_dur)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_ylim(0, 1.1*np.max(const_fa))
# ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_T1T2[0], True)
plt.title('Low $cost_{T1,T2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_T1T2[500000], True)
plt.title('Medium $cost_{T1,T2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

index = -1
while True:
    if np.count_nonzero(seqs_sorted_T1T2[index].prep==1) > 0 and np.count_nonzero(seqs_sorted_T1T2[index].prep==2) > 0:
        break
    else:
        index -= 1

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_T1T2[index], True)
plt.title('High $cost_{T1,T2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(mrf_sequence_hamilton, True)
plt.title('Hamilton')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(mrf_sequence_jaubert, True)
plt.title('Jaubert')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(mrf_sequence_kvernby, True)
plt.title('Kvernby')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_ylim(0, 1.1*np.max(const_fa))
ax.set_xlabel('Time [ms]')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend([list(by_label.values())[2], list(by_label.values())[0], list(by_label.values())[1]], [list(by_label.keys())[2], list(by_label.keys())[0], list(by_label.keys())[1]], loc='upper center', ncols=3, bbox_to_anchor=(0.5, 17))

fig.subplots_adjust(hspace=1)

# plt.savefig(f'/home/tomgr/Documents/abdominal/figures/sequences_{timestamp}.png', dpi=300, bbox_inches='tight')
# %%
