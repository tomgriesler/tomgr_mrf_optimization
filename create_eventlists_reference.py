#%%
import numpy as np
from pathlib import Path
import shutil

from abdominal_tools import MRFSequence, BLOCKS, visualize_sequence

# %%
prep_blocks_dict = {
    'jaubert': ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI300',     'noPrep', 'T2prep40', 'T2prep80', 'T2prep120', 'TI12', 'noPrep'],
    'kvernby': ['T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50', 'TI100', 'noPrep', 'noPrep', 'noPrep', 'T2prep50'],
    'hamilton': ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']
}
#%%
which = 'hamilton'
beats = 16
shots = 35
const_tr = 5.7
te = 1.
prep_blocks = prep_blocks_dict[which]
phase_inc = 3.
total_dur = 1e4
name = f'{which}_{total_dur/1e3:.0f}s_{beats}_phinc{phase_inc:.0f}deg'

#%%
prep_order = np.concatenate((np.tile(prep_blocks, reps=beats//len(prep_blocks)), prep_blocks[:beats%len(prep_blocks)]))
waittimes = np.full(beats-1, total_dur - np.sum([BLOCKS[prep]['ti'] + BLOCKS[prep]['t2te'] + shots*const_tr for prep in prep_order]))/(beats-1)
prep = [BLOCKS[name]['prep'] for name in prep_order]
ti = [BLOCKS[name]['ti'] for name in prep_order]
t2te = [BLOCKS[name]['t2te'] for name in prep_order]
fa = np.full(beats * shots, 15.)
tr = np.full(beats*shots, 0)
for ii in range(len(waittimes)):
    tr[(ii+1)*shots-1] += waittimes[ii]*1e3
ph = phase_inc*np.arange(beats*shots).cumsum()
mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, const_tr, te)

#%%
visualize_sequence(mrf_seq, show_fa=True)

#%% save lists
savepath = Path(f'/home/tomgr/Documents/abdominal/data/sequences/{name}/textfiles_{name}')
savepath.mkdir(exist_ok=True, parents=True)
np.savetxt(savepath/'PREP_FISP.txt', mrf_seq.prep, fmt='%i')
np.savetxt(savepath/'TI_FISP.txt', mrf_seq.ti, fmt='%f')
np.savetxt(savepath/'T2TE_FISP.txt', mrf_seq.t2te, fmt='%f')
np.savetxt(savepath/'FA_FISP.txt', mrf_seq.fa, fmt='%f')
np.savetxt(savepath/'TR_FISP.txt', mrf_seq.tr, fmt='%f')
np.savetxt(savepath/'PH_FISP.txt', mrf_seq.ph, fmt='%f')
shutil.copyfile('/home/tomgr/Documents/abdominal/code/reconstruction/ID_FISP.txt', savepath/'ID_FISP.txt')

# %%
