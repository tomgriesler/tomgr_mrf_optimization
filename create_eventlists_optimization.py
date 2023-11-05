#%%
import numpy as np
from pathlib import Path
import pickle
import json

from abdominal_tools import TargetTissue, RESULTSPATH, visualize_sequence, sort_sequences, create_weightingmatrix

#%%
timestamp = '231105_070955'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

with open(resultspath/'acq_block.pkl', 'rb') as handle: 
    acq_block = pickle.load(handle)

with open(resultspath/'prot.json', 'r') as handle: 
    prot = json.load(handle)

target_tissue = TargetTissue(prot['target_tissue']['T1'], prot['target_tissue']['T2'], prot['target_tissue']['M0'])
inversion_efficiency = prot['inversion_efficiency']
delta_B1 = prot['delta_B1']


#%%
weighting = '1/T1, 0, 0'
weighting = '0, 1/T2, 0'
weighting = '1/T1, 1/T2, 0'

#%%
weightingmatrix = create_weightingmatrix(target_tissue, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
mrf_seq = sequences[-2]
name = timestamp + '_worst_1_T1T2'

#%%
waittimes = mrf_seq.waittimes

#%%
visualize_sequence(mrf_seq, acq_block)

# %%
FA_FISP = np.tile(acq_block.fa, len(mrf_seq.PREP))

TR_FISP = np.zeros_like(FA_FISP)
for i, waittime in enumerate(waittimes):
    TR_FISP[len(acq_block.tr)*(i+1)-1] = waittime * 1e3

PH_FISP = np.zeros_like(FA_FISP)

#%% save lists
savepath = Path(f'/home/tomgr/Documents/shared_files/{name}')
savepath.mkdir(exist_ok=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.PREP, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.TI, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.T2TE, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', FA_FISP, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', TR_FISP, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', PH_FISP, fmt='%f')

# %%
ii = -65
while True:
    if 1 in sequences[ii].PREP and 2 in sequences[ii].PREP:
        print(ii)
        break
    else:
        ii -= 1
# %%
