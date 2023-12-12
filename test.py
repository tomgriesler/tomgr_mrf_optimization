#%%
import numpy as np
import matplotlib.pyplot as plt

from abdominal_tools import MRFSequence, visualize_sequence, create_weightingmatrix
from signalmodel_abdominal import calculate_orthogonality, calculate_signal
#%%
beats = 15
shots = 47
n_ex = beats * shots
fa = np.loadtxt('/home/tomgriesler/Documents/UM/code/abdominal/textfiles/FA_FISP_sydney.txt')
tr = np.zeros(n_ex)
tr_offset = 5.4
ph = np.zeros(n_ex)

# # Jesse's cardiac T1T2 sequence
# prep = [1, 0, 2, 2] * 4
# ti = [21, 0, 0, 0, 100, 0, 0, 0, 250, 0, 0, 0, 400, 0, 0, 0]
# t2te = [0, 0, 40, 80] * 4

# Sydney's cardiac T1T2T1rho sequence
prep = [1, 0, 3, 3, 3] + [1, 0, 2, 2, 2] * 2
ti = [21, 0, 0, 0, 0] * 3
t2te = [0, 0, 0, 0, 0] + [0, 0, 30, 50, 80] * 2
tsl = [0, 0, 30, 50, 60] + [0] * 10

te = 1.

# for ii in range(1, beats):
#     tr[ii*shots-1] += (total_dur - np.sum(ti) - np.sum(t2te) - beats*shots*tr_offset)*1e3/(beats-1)

for ii in range(beats):
    tr[(ii+1)*shots-1] += 1e6 - (ti[ii] + t2te[ii] + tsl[ii] + tr_offset*shots)*1e3

costfunction = 'crlb'

# Relaxation times of healthy myocardium
t1 = 1000
t2 = 44
m0 = 1
t1rho = 50

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
mrf_seq.calc_cost(costfunction, t1, t2, m0)
print(mrf_seq.cost)

#%%
mrf_seq.calc_cost(costfunction, t1, t2, m0, t1rho=t2)

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
    mrf_seq.calc_cost(costfunction, t1, t2, m0)
    # print(phase_inc, mrf_seq.cost, np.sum(np.multiply(weightingmatrix, mrf_seq.cost)))
    w_crlb = np.multiply(weightingmatrix, mrf_seq.cost)
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
calculate_orthogonality([150, 828], [20, 72], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)
# %%
beats = 1

fa = np.load('/home/tomgriesler/Documents/Uni_Wue/Master/tg_mrf_optimization/initialization/fa_cao.npy')
tr = np.zeros_like(fa)

shots = len(fa)
ph = np.zeros_like(fa)
prep = [1]
ti = [20]
t2te = [0]

tr_offset = 12

signal_1 = calculate_signal(150, 20, 1, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)
signal_2 = calculate_signal(828, 72, 1, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)

plt.plot(-np.imag(signal_1))
plt.plot(-np.imag(signal_2))

# %%
calculate_orthogonality([150, 828], [20, 72], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)
# %%
