#%%
import numpy as np
import matplotlib.pyplot as plt

from utils.abdominal_tools import MRFSequence

#%%
beats = 1

fa = np.load('/home/tomgr/Documents/misc_code/tg_mrf_optimization/initialization/fa_cao.npy')
tr = np.zeros_like(fa)

shots = len(fa)

prep = [1]
ti = [20]
t2te = [0]
ph = np.zeros_like(fa)
ph[::2] = 180

tr_offset = 12.

mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, 1.4)

#%%
t1 = 660.
t2 = 40.
m0 = 1.

# %%
signal_fisp = mrf_seq.calc_signal_fisp(t1, t2, m0, inv_eff=1., return_result=True)
signal_bssfp = mrf_seq.calc_signal_bssfp(t1, t2, m0, inv_eff=1., return_result=True)

plt.plot(-np.imag(signal_fisp), label='fisp')
plt.plot(-np.imag(signal_bssfp), label='bssfp')

plt.legend()
# %%
