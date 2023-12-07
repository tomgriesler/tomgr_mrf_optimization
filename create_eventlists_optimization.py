#%%
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm

from abdominal_tools import RESULTSPATH, visualize_sequence, sort_sequences, create_weightingmatrix

#%%
timestamp = '231204_090948'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

#%%
for sequence in tqdm(sequences, total=len(sequences), desc='Decompressing'):
    sequence.decompress()

#%% old implementation
target_t1 = prot['target_t1']
target_t2 = prot['target_t2']
target_m0 = prot['target_m0']
shots = prot['shots']
const_fa = prot['const_fa']
const_tr = prot['const_tr']
te = prot['te']
total_dur = prot['total_dur']
phase_inc = prot['phase_inc']
inv_eff = prot['inversion_efficiency']
delta_B1 = prot['delta_B1']

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

#%%
weighting = '1/T1, 0, 0'
weighting = '0, 1/T2, 0'
weighting = '1/T1, 1/T2, 0'

#%%
weightingmatrix = create_weightingmatrix(target_t1, target_t2, target_m0, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
mrf_seq = sequences[0]
name = timestamp + '_best_T1T2'

#%%
visualize_sequence(mrf_seq, show_fa=True)

#%% save lists
savepath = Path(f'/home/tomgr/Documents/abdominal/data/sequences/{timestamp}/{name}/textfiles_{name}')
savepath.mkdir(exist_ok=True, parents=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.prep, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.ti, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.t2te, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', mrf_seq.fa, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', mrf_seq.tr, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', mrf_seq.ph, fmt='%f')

# %%
