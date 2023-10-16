#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json

from abdominal_tools import MRFSequence, AcquisitionBlock, TargetTissue, BLOCKS, RESULTSPATH, visualize_sequence, sort_sequences, create_weightingmatrix
# %%
prep_order = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep']

prep_order = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']

prep_order = ['noPrep']

name = 'optim_T2_10s'


#%%
total_dur = 1e4

acq_block = AcquisitionBlock(np.full(40, 15.), np.full(40, 5.), TE=2.71)

waittimes = [total_dur/len(prep_order) - BLOCKS[name]['ti']-BLOCKS[name]['t2te']-sum(acq_block.tr) for name in prep_order]

mrf_seq = MRFSequence(prep_order, waittimes)

#%%
timestamp = '231006_180622'
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
weighting = '0, 1/T2, 0'
weightingmatrix = create_weightingmatrix(target_tissue, weighting)
sort_sequences(sequences, weightingmatrix)

#%%
mrf_seq = sequences[int(len(sequences)/2)+1]
mrf_seq = sequences[0]

#%%
waittimes = mrf_seq.waittimes
#%%
visualize_sequence(mrf_seq, acq_block)
# %%
FA_FISP = np.tile(acq_block.fa, len(mrf_seq.PREP))

TR_FISP = np.tile(acq_block.tr, len(mrf_seq.PREP))
for i, waittime in enumerate(waittimes):
    TR_FISP[len(acq_block.tr)*(i+1)-1] += waittime

PH_FISP = np.zeros_like(FA_FISP)

#%% save lists
savepath = Path(f'/home/tomgr/Documents/MRF_sim_for_tom/Sequences/{name}')
savepath.mkdir(exist_ok=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.PREP, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.TI, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.T2TE, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', FA_FISP, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', TR_FISP, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', PH_FISP, fmt='%f')


#%%
target_tissue = TargetTissue(660, 40, 1)

mrf_seq.calc_signal(acq_block, target_tissue, inversion_efficiency=0.95)

plt.plot(-np.imag(mrf_seq.signal)/np.linalg.norm(np.imag(mrf_seq.signal)))

# plt.ylim(0.12, 0.28)

plt.show()

#%%
np.savetxt(savepath/f'signal_{target_tissue.T1}_{target_tissue.T2}.txt', -np.imag(mrf_seq.signal)/np.linalg.norm(np.imag(mrf_seq.signal)))


# %%
