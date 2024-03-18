#%%
import numpy as np
import matplotlib.pyplot as plt

from utils.abdominal_tools import MRFSequence
from utils.visualization import visualize_sequence

#%%
beats = 1

fa = np.load('/home/tomgr/Documents/misc_code/tg_mrf_optimization/initialization/fa_cao.npy')
tr = np.zeros_like(fa)

shots = len(fa)

prep = [1]
ti = [20]
t2te = [0]
ph = np.zeros_like(fa)

tr_offset = 12.
ph[::2] = 180

# %% not alternating
mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, 1.4)

plt.figure()

for t1, t2 in zip([4000, 800, 150], [600, 70, 20]):
    signal = mrf_seq.calc_signal_fisp(t1, t2, 1, inv_eff=1., return_result=True)
    plt.plot(-np.imag(signal), label=f'%i, %i' %(t1, t2))

plt.legend()
# %%
