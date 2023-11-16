#%%
import numpy as np
import matplotlib.pyplot as plt

from abdominal_tools import MRFSequence

#%%
beats = 16
shots = 35
n_ex = beats*shots
fa = np.full(n_ex, 15)
tr = np.zeros(n_ex)
tr_offset = 5
ph = np.zeros(n_ex)
prep = [1, 0, 2, 2] * 4
ti = [21, 0, 0, 0, 100, 0, 0, 0, 250, 0, 0, 0, 400, 0, 0, 0]
t2te = [0, 0, 40, 80] * 4
te = 1.
total_dur = 1e4

for ii in range(beats):
    tr[(ii+1)*shots-1] += (total_dur - np.sum(ti) - np.sum(t2te) - beats*shots*tr_offset)*1e3/beats

t1 = 660
t2 = 40
m0 = 1

#%%
mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te)
mrf_seq.calc_signal(t1, t2, m0)

#%%
plt.plot(np.real(mrf_seq.signal))
plt.plot(np.imag(mrf_seq.signal))

# %%
mrf_seq.calc_crlb(t1, t2, m0)
print(mrf_seq.crlb)
# %%
