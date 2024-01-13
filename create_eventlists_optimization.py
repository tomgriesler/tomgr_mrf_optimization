#%%
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import shutil

from abdominal_tools import RESULTSPATH, visualize_sequence, sort_sequences, create_weightingmatrix

#%%
timestamp = '231218_074418'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

#%%
for sequence in tqdm(sequences, total=len(sequences), desc='Decompressing'):
    sequence.decompress()

#%% old implementation
# target_t1 = prot['target_t1']
# target_t2 = prot['target_t2']
# target_m0 = prot['target_m0']
# shots = prot['shots']
# const_fa = prot['const_fa']
# const_tr = prot['const_tr']
# te = prot['te']
# total_dur = prot['total_dur']
# phase_inc = prot['phase_inc']
# inv_eff = prot['inversion_efficiency']
# delta_B1 = prot['delta_B1']

#%% new implementation
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

try: target_t1rho = prot['target_t1rho'] 
except KeyError: pass

#%%
weighting = 'T1'
weighting = 'T2'
weighting = 'T1rho'
weighting = 'T1, T2'
weighting = 'T1, T2, T1rho'

#%%
if costfunction == 'crlb':
    weightingmatrix = create_weightingmatrix(weighting, target_t1, target_t2, target_t1rho, dims=4)
    seqs_sorted = sort_sequences(sequences, weightingmatrix)
elif costfunction == 'orthogonality' or 'crlb_orth':
    seqs_sorted = sort_sequences(sequences)

#%%
mrf_seq = seqs_sorted[0]

#%%
name = timestamp + '_best_T1rho'
name = timestamp + '_best_T1T2T1rho'

#%%
print(np.multiply(weightingmatrix, mrf_seq.cost))
print(np.sum(np.multiply(weightingmatrix, mrf_seq.cost)))
visualize_sequence(mrf_seq, show_fa=True)

#%% save lists
savepath = Path(f'/home/tomgr/Documents/abdominal/data/sequences/{timestamp}/{name}/textfiles_{name}')
savepath.mkdir(exist_ok=True, parents=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.prep, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.ti, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.t2te+mrf_seq.tsl, fmt='%f')
# np.savetxt(savepath/'TSL_FISP.txt', mrf_seq.tsl, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', mrf_seq.fa, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', mrf_seq.tr, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', mrf_seq.ph, fmt='%f')
shutil.copyfile('/home/tomgr/Documents/abdominal/code/reconstruction/ID_FISP.txt', savepath/'ID_FISP.txt')

# # %% (Sydney's naming convention)
# savepath = Path(f'/home/tomgr/Documents/abdominal/data/sequences/{timestamp}/sydney/{name}/textfiles_{name}')
# savepath.mkdir(exist_ok=True, parents=True)
# np.savetxt(savepath/'PREP_FISP_T2.txt', mrf_seq.prep, fmt='%i')
# np.savetxt(savepath/'TI_FISP.txt', mrf_seq.ti, fmt='%f')
# np.savetxt(savepath/'T2T1rhoPREPtime_MRF.txt', (mrf_seq.t2te+mrf_seq.tsl)*1e3, fmt='%f')
# np.savetxt(savepath/'FA_FISP.txt', mrf_seq.fa, fmt='%f')
# np.savetxt(savepath/'TR_FISP.txt', mrf_seq.tr, fmt='%f')
# np.savetxt(savepath/'PH_FISP.txt', mrf_seq.ph, fmt='%f')
# shutil.copyfile('/home/tomgr/Documents/abdominal/code/reconstruction/ID_FISP.txt', savepath/'ID_FISP.txt')
# # %%

# %%
