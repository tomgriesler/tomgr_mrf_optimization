#%%
import numpy as np
import matplotlib.pyplot as plt

from abdominal_tools import MRFSequence

#%%
beats = 1

fa = np.load('/home/tomgr/Documents/code/tg_mrf_optimization/initialization/fa_cao.npy')
tr = np.zeros_like(fa)

shots = len(fa)

prep = [1]
ti = [20]
t2te = [0]
ph = np.zeros_like(fa)

tr_offset = 12.

mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, 1.4)
# %%
t1 = 660.
t2 = 40.
m0 = 1.

mrf_seq.calc_crlb(t1, t2, m0, inversion_efficiency=1.)
print(mrf_seq.crlb)
# %%
mrf_seq.calc_signal(t1, t2, m0, inversion_efficiency=1.)
plt.plot(-np.imag(mrf_seq.signal))

# %%
