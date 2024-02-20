#%%
import pickle
import json
import numpy as np
from tqdm import tqdm

from utils.abdominal_tools import RESULTSPATH, create_weightingmatrix, sort_sequences, visualize_sequence

#%%
timestamp = '231210_034637'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

#%%
for sequence in tqdm(sequences, total=len(sequences), desc='Decompressing'):
    sequence.decompress()

#%%
target_t1 = prot['target_t1']
target_t2 = prot['target_t2']
try:
    target_t1rho = prot['target_t1rho']
    dims = 4
except KeyError:
    target_t1rho = None
    dims = 3

#%%
weighting = 'T1'
weightingmatrix_T1 = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims)
seqs_sorted_T1 = sort_sequences(sequences, weightingmatrix_T1)

weighting = 'T2'
weightingmatrix_T2 = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims)
seqs_sorted_T2 = sort_sequences(sequences, weightingmatrix_T2)

weighting = 'T1, T2'
weightingmatrix_T1T2 = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims)
seqs_sorted_T1T2 = sort_sequences(sequences, weightingmatrix_T1T2)

weighting = 'T1rho'
weightingmatrix_T1rho = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims)
seqs_sorted_T1rho = sort_sequences(sequences, weightingmatrix_T1rho)

weighting = 'T1, T2, T1rho'
weightingmatrix_T1T2T1rho = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims)
seqs_sorted_T1T2T1rho = sort_sequences(sequences, weightingmatrix_T1T2T1rho)

#%%
N = 100

#%%
weightings = ['T1', 'T2', 'T1rho', 'all']
seqs_sorted = [seqs_sorted_T1[:N], seqs_sorted_T2[:N], seqs_sorted_T1rho[:N], seqs_sorted_T1T2T1rho[:N]]

results = {name: {} for name in weightings}

for name, seqs in tqdm(zip(weightings, seqs_sorted), total=len(weightings)):
    results[name]['fa'] = np.mean([np.mean(seq.fa) for seq in seqs])
    results[name]['n'] = np.mean([len(seq.prep) for seq in seqs])
    results[name]['n_1'] = np.mean([np.count_nonzero(seq.prep==1) for seq in seqs])
    results[name]['n_2'] = np.mean([np.count_nonzero(seq.prep==2) for seq in seqs])
    results[name]['n_3'] = np.mean([np.count_nonzero(seq.prep==3) for seq in seqs])


# %%
print('\t', end='')
for what in results[weightings[0]].keys(): print(what, end='\t')
for sortby in results.keys():
    print('\n', sortby, end='\t')
    for what in results[sortby].keys():
        print(f'{results[sortby][what]:.2f}', end='\t')
# %%
