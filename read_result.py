#%%
import pickle
import matplotlib.pyplot as plt
import json

# %%
from abdominal_tools import RESULTSPATH, MRFSequence
#%%
timestamp = '230927_1335'
resultspath = RESULTSPATH/timestamp

#%%
with open(resultspath/'best_sequences.pkl', 'rb') as handle:
    best_sequences = pickle.load(handle).values()
with open(resultspath/'mrf_sequence_ref.pkl', 'rb') as handle:
    mrf_sequence_ref = pickle.load(handle)
with open(resultspath/'mrf_sequence_best.pkl', 'rb') as handle:
    mrf_sequence_best = pickle.load(handle)
with open(resultspath/'mrf_sequence_worst.pkl', 'rb') as handle:
    mrf_sequence_worst = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle:
    prot = json.load(handle)

# %%
plt.plot(sorted(prot['crlbs']), label='total')
plt.plot(prot['crlbs_T1'], label='T1')
plt.plot(prot['crlbs_T2'], label='T2')
plt.axhline(mrf_sequence_ref.crlb, ls='--')
plt.axhline(mrf_sequence_ref.crlb_T1, ls='--', color='tab:orange')
plt.axhline(mrf_sequence_ref.crlb_T2, ls='--', color='tab:green')
# %%
