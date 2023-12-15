#%%
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

from abdominal_tools import RESULTSPATH, BLOCKS, visualize_sequence, visualize_cost,create_weightingmatrix,sort_sequences, MRFSequence

#%%
timestamp = '231215_083347'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

#%%
for sequence in tqdm(sequences, total=len(sequences), desc='Decompressing'):
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

try:
    target_t1rho = prot['target_t1rho']
except KeyError:
    target_t1rho = None

#%% Compare to reference
beats_jaubert = 16
beats_kvernby = 16
beats_hamilton = 16

prep_blocks_jaubert = ['TI12', 'noPrep', 't2prep40', 't2prep80', 't2prep120', 'TI300', 'noPrep', 't2prep40', 't2prep80', 't2prep120', 'TI12', 'noPrep']
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

prep_blocks_kvernby = ['t2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 't2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 't2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 't2prep50']
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

prep_blocks_hamilton = ['TI21', 'noPrep', 't2prep40', 't2prep80', 'TI100', 'noPrep', 't2prep40', 't2prep80', 'TI250', 'noPrep', 't2prep40', 't2prep80', 'TI400', 'noPrep', 't2prep40', 't2prep80']
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

mrf_sequence_jaubert.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)
mrf_sequence_kvernby.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)
mrf_sequence_hamilton.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)

#%%
print('Sorting by cost_{t1,t2}...', end='')
weighting = 't1, t2'
weightingmatrix_t1t2 = create_weightingmatrix(weighting, target_t1, target_t2, dims=3)
seqs_sorted_t1t2 = sort_sequences(sequences, weightingmatrix_t1t2)
print('done.')

print('Sorting by cost_{t1}...', end='')
weighting = 'T1'
weightingmatrix_t1 = create_weightingmatrix(weighting, target_t1, target_t2, dims=3)
seqs_sorted_t1 = sort_sequences(sequences, weightingmatrix_t1)
print('done.')

print('Sorting by cost_{t2}...', end='')
weighting = 'T2'
weightingmatrix_t2 = create_weightingmatrix(weighting, target_t1, target_t2, dims=3)
seqs_sorted_t2 = sort_sequences(sequences, weightingmatrix_t2)
print('done.')

#%%
prep_sydney = [1, 0, 3, 3, 3] + [1, 0, 2, 2, 2] * 2
ti_sydney = [21, 0, 0, 0, 0] * 3
t2te_sydney = [0, 0, 0, 0, 0] + [0, 0, 30, 50, 80] * 2
tsl_sydney = [0, 0, 30, 50, 60] + [0] * 10
beats_sydney = 15
n_ex_sydney = beats_sydney * shots
fa_sydney = np.loadtxt('/home/tomgr/Documents/abdominal/textfiles/FA_FISP_sydney.txt')
tr_sydney = np.zeros(n_ex_sydney)
tr_offset_sydney = 5.4
# ph_sydney = np.zeros(n_ex_sydney)
ph_sydney = phase_inc * np.arange(n_ex_sydney).cumsum()
for ii in range(beats_sydney):
    tr_sydney[(ii+1)*shots-1] += 1e6 - (ti_sydney[ii] + t2te_sydney[ii] + tsl_sydney[ii] + tr_offset_sydney*shots)*1e3
mrf_sequence_sydney = MRFSequence(beats_sydney, shots, fa_sydney, tr_sydney, ph_sydney, prep_sydney, ti_sydney, t2te_sydney, tr_offset_sydney, te, tsl_sydney)

mrf_sequence_sydney.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1, t1rho=target_t1rho)

#%%
print('Sorting by cost_{t1, t2, t1rho}.')
weighting = 'T1, T2, T1rho'
weightingmatrix_t1t2t1rho = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims=4)
seqs_sorted_t1t2t1rho = sort_sequences(sequences, weightingmatrix_t1t2t1rho)

print('Sorting by cost_{t1rho}.')
weighting = 'T1rho'
weightingmatrix_t1rho = create_weightingmatrix(weighting, target_t1rho=target_t1rho, dims=4)
seqs_sorted_t1rho = sort_sequences(sequences, weightingmatrix_t1rho)

#%%
print('Sorting by orthogonality...', end='')
seqs_sorted_orth = sort_sequences(sequences)
print('done.')

# %%
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(16, 9))

plt.subplot(1, 3, 1)
visualize_cost(seqs_sorted_t1, weightingmatrix_t1t2)
plt.axhline(np.sum(np.multiply(weightingmatrix_t1, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{1, ref}$', color='tab:blue', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
plt.ylabel('cost function value')
plt.title('Sorted by $cost_1$')

plt.subplot(1, 3, 2)
visualize_cost(seqs_sorted_t2, weightingmatrix_t1t2)
plt.axhline(np.sum(np.multiply(weightingmatrix_t2, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{2, ref}$', color='tab:red', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
ax = plt.gca()
ax.set_yticklabels([])
plt.title('Sorted by $cost_2$')

plt.subplot(1, 3, 3)
visualize_cost(seqs_sorted_t1t2, weightingmatrix_t1t2)
plt.axhline(np.sum(np.multiply(weightingmatrix_t1t2, mrf_sequence_hamilton.cost)), ls=':', label='$cost_{3, ref}$', color='tab:green', linewidth=2)
plt.xlim(0, len(sequences))
plt.ylim(0, 10)
plt.legend(loc='upper left', markerscale=200)
plt.xlabel('Index')
ax = plt.gca()
ax.set_yticklabels([])
plt.title('Sorted by $cost_3$')

plt.tight_layout()

plt.show()

# %%
plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(16, 9))

n_subplots = 8
ii = 1

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_t1[0], True)
plt.title('Low $cost_{t1}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

# plt.subplot(n_subplots, 1, ii)
# visualize_sequence(seqs_sorted_t1[-1])
# plt.title('High $cost_{t1}$')
# plt.xlim(0, total_dur)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_ylim(0, 1.1*np.max(const_fa))
# ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_t2[0], True)
plt.title('Low $cost_{t2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

# plt.subplot(n_subplots, 1, ii)
# visualize_sequence(seqs_sorted_t2[-1])
# plt.title('High $cost_{t2}$')
# plt.xlim(0, total_dur)
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_ylim(0, 1.1*np.max(const_fa))
# ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_t1t2[0], True)
plt.title('Low $cost_{t1,t2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_t1t2[500000], True)
plt.title('Medium $cost_{t1,t2}$')
plt.xlim(0, total_dur)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_ylim(0, 1.1*np.max(const_fa))
ii += 1 

index = -1
while True:
    if np.count_nonzero(seqs_sorted_t1t2[index].prep==1) > 0 and np.count_nonzero(seqs_sorted_t1t2[index].prep==2) > 0:
        break
    else:
        index -= 1

plt.subplot(n_subplots, 1, ii)
visualize_sequence(seqs_sorted_t1t2[index], True)
plt.title('High $cost_{t1,t2}$')
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

# %%
fig = plt.figure(figsize=(16, 9))

for ii in range(10):
    plt.subplot(10, 1, ii+1)
    visualize_sequence(seqs_sorted_t1t2t1rho[ii], True)
    # %%
for ii in range(10):
    print(np.sum(np.multiply(weightingmatrix_t1t2, seqs_sorted_t1t2[ii].cost)))

# %%
_, ax = plt.subplots(2, 2)
for t1, t2, t1rho in zip(target_t1, target_t2, target_t1rho):
    signal_ref = mrf_sequence_sydney.calc_signal(t1, t2, target_m0, inv_eff, delta_B1, t1rho, True)
    signal_opt = seqs_sorted_orth[3].calc_signal(t1, t2, target_m0, inv_eff, delta_B1, t1rho, True)
    ax[0, 0].plot(np.real(signal_ref))
    ax[1, 0].plot(np.real(signal_opt))
    ax[0, 1].plot(np.imag(signal_ref))
    ax[1, 1].plot(np.imag(signal_opt))
    
# %%
