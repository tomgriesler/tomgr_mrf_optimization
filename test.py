#%%
import numpy as np
import matplotlib.pyplot as plt

from abdominal_tools import MRFSequence, visualize_sequence, create_weightingmatrix

#%%
beats = 16
shots = 35
n_ex = beats*shots
fa = np.full(n_ex, 15)
tr = np.zeros(n_ex)
tr_offset = 5
ph = np.zeros(n_ex)
# ph = np.tile(4*np.arange(shots).cumsum(), beats)
prep = [1, 0, 2, 2] * 4
ti = [21, 0, 0, 0, 100, 0, 0, 0, 250, 0, 0, 0, 400, 0, 0, 0]
t2te = [0, 0, 40, 80] * 4
te = 1.
total_dur = 1e4

for ii in range(1, beats):
    tr[ii*shots-1] += (total_dur - np.sum(ti) - np.sum(t2te) - beats*shots*tr_offset)*1e3/(beats-1)

t1 = 660
t2 = 40
m0 = 1

#%%
mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te)

#%%
visualize_sequence(mrf_seq)

#%%
mrf_seq.calc_signal(t1, t2, m0)

#%%
plt.plot(np.real(mrf_seq.signal))
plt.plot(np.imag(mrf_seq.signal))

# %%
mrf_seq.calc_crlb(t1, t2, m0)
print(mrf_seq.crlb)


#%%
weightingmatrix = create_weightingmatrix(t1, t2, m0, '1/T1, 1/T2, 0')
res = []
res_t1 = []
res_t2 = []

for phase_inc in np.arange(0, 10):
    # ph = np.tile(phase_inc*np.arange(shots).cumsum(), beats)
    ph = phase_inc*np.arange(beats*shots).cumsum()
    # ph = phase_inc*np.arange(beats*shots)
    # ph = phase_inc*np.sin(np.arange(beats*shots)*4*np.pi/beats/shots)
    mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te)
    mrf_seq.calc_crlb(t1, t2, m0)
    # print(phase_inc, mrf_seq.crlb, np.sum(np.multiply(weightingmatrix, mrf_seq.crlb)))
    w_crlb = np.multiply(weightingmatrix, mrf_seq.crlb)
    res.append(np.sum(w_crlb))
    res_t1.append(w_crlb[0])
    res_t2.append(w_crlb[1])

#%%
plt.plot(res, label='cost$_{T1,T2}$:\t' + f'-{(1-min(res)/res[0])*100:.1f}%')
plt.plot(res_t1, label='cost$_{T1}$:\t' + f'-{(1-min(res_t1)/res_t1[0])*100:.1f}%')
plt.plot(res_t2, label='cost$_{T2}$:\t' + f'-{(1-min(res_t2)/res_t2[0])*100:.1f}%')
plt.ylim(0, 4)
plt.xlabel('$\Delta \Phi$ [deg]')
plt.ylabel('CRLB')
plt.legend()
# %%
print(1-min(res)/res[0])
# %%
