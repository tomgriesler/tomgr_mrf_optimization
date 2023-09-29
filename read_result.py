#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
# %%
from abdominal_tools import RESULTSPATH, visualize_sequence
#%%
timestamp = '230927_1752'
timestamp = '230927_1753'
timestamp = '230928_0905'

#%%
resultspath = RESULTSPATH/timestamp

#%%
with open(resultspath/'best_sequences.pkl', 'rb') as handle:
    best_sequences = list(pickle.load(handle).values())
with open(resultspath/'worst_sequences.pkl', 'rb') as handle:
    worst_sequences = list(pickle.load(handle).values())
with open(resultspath/'mrf_sequence_ref.pkl', 'rb') as handle:
    mrf_sequence_ref = pickle.load(handle)
crlb_array = np.load(resultspath/'crlb_array.npy')
with open(resultspath/'prot.json', 'r') as handle:
    prot = json.load(handle)

# %%
fig = plt.figure(figsize=(16, 9))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    visualize_sequence(best_sequences[i])
    ax.set_title(f'CRLB={best_sequences[i].crlb:.3f}')
plt.tight_layout()

# %%
fig = plt.figure(figsize=(16, 9))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    visualize_sequence(worst_sequences[-6+i])
    ax.set_title(f'CRLB={worst_sequences[-6+i].crlb:.3f}')
plt.tight_layout()

# %%
for i in range(4):
    plt.plot(crlb_array[i], '.', ms=1, label=['tot', 'T1', 'T2', 'M0'][i])
plt.ylim(0, 2000)
plt.axhline(mrf_sequence_ref.crlb, label='reference', ls='--')
plt.axhline(mrf_sequence_ref.crlb_T1, color='tab:orange', ls='--')
plt.axhline(mrf_sequence_ref.crlb_T2, color='tab:green', ls='--')
plt.axhline(mrf_sequence_ref.crlb_M0, color='tab:red', ls='--')
plt.legend()
plt.title(f'{len(mrf_sequence_ref.prep_order)} Blocks, Reference: {prot["reference"].capitalize()}')
plt.show()
# %%
