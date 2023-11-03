#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json

from abdominal_tools import MRFSequence, AcquisitionBlock, TargetTissue, BLOCKS, RESULTSPATH, visualize_sequence, sort_sequences, create_weightingmatrix
# %%
# Jaubert mod 4
name = 'jaubert_mod_10s_4'
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80']

# Jaubert mod 8
name = 'jaubert_mod_10s_8'
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40']

# Jaubert mod 12
name = 'jaubert_mod_10s_12'
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep']

# Jaubert mod 16
name = 'jaubert_mod_10s_16'
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300']

# Jaubert mod 20
name = 'jaubert_mod_10s_20'
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120']

# Hamilton 4
name = 'hamilton_10s_4'
prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80']

# Hamilton 8
name = 'hamilton_10s_8'
prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80']

# Hamilton 12
name = 'hamilton_10s_12'
prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80']

# Hamilton 16
name = 'hamilton_10s_16'
prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

# Hamilton 20
name = 'hamilton_10s_20'
prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80', 'TI21', 'noPrep', 'T2prep40', 'T2prep80']

# 3D QALAS 4
name = '3dqalas_10s_4'
prep_order = ['T2prep50', 'TI100', 'noPrep', 'noPrep']

# 3D QALAS 8
name = '3dqalas_10s_8'
prep_order = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep']

# 3D QALAS 12
name = '3dqalas_10s_12'
prep_order = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100']

# 3D QALAS 16
name = '3dqalas_10s_16'
prep_order = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50']

# 3D QALAS 20
name = '3dqalas_10s_20'
prep_order = ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep']


#%%
total_dur = 1e4

# acq_block = AcquisitionBlock(np.full(40, 15.), np.full(40, 5.), TE=1.4)
acq_block = AcquisitionBlock(np.full(35, 15.), np.full(35, 5.7), TE=1.4)

waittimes = np.concatenate((np.full(len(prep_order)-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + sum(acq_block.tr) for prep in prep_order]))/(len(prep_order)-1), [0]))

mrf_seq = MRFSequence(prep_order, waittimes)

#%%
timestamp = '231020_074613'
resultspath = RESULTSPATH/timestamp

with open(resultspath/'sequences.pkl', 'rb') as handle: 
    sequences = pickle.load(handle)

# Load acq block from optimization folder
# with open(resultspath/'acq_block.pkl', 'rb') as handle: 
#     acq_block = pickle.load(handle)
# acq_block = AcquisitionBlock(np.full(37, 15.), np.full(37, 5.4), 1.4)

# Create custom acq block
acq_block = AcquisitionBlock(np.full(35, 15.), np.full(35, 5.7), 1.4)

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
mrf_seq = sequences[-91]

#%%
waittimes = mrf_seq.waittimes

#%%
visualize_sequence(mrf_seq, acq_block)

#%%
name = timestamp + '_worst_T1T2_35'

# %%
FA_FISP = np.tile(acq_block.fa, len(mrf_seq.PREP))

# TR_FISP = np.tile(acq_block.tr, len(mrf_seq.PREP))
# for i, waittime in enumerate(waittimes):
#     TR_FISP[len(acq_block.tr)*(i+1)-1] += waittime

TR_FISP = np.zeros_like(FA_FISP)
for i, waittime in enumerate(waittimes):
    # TR_FISP[len(acq_block.tr)*(i+1)-1] = (waittime + 0.2) * 1e3
    TR_FISP[len(acq_block.tr)*(i+1)-1] = (waittime + 0.5) * 1e3

PH_FISP = np.zeros_like(FA_FISP)

#%% save lists
# savepath = Path(f'/home/tomgr/Documents/MRF_sim_for_tom/Sequences/{name}')
savepath = Path(f'/home/tomgr/Documents/shared_files/{name}')
savepath.mkdir(exist_ok=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.PREP, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.TI, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.T2TE, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', FA_FISP, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', TR_FISP, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', PH_FISP, fmt='%f')
