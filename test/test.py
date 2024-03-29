#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.abdominal_tools import MRFSequence, create_weightingmatrix
from utils.visualization import visualize_sequence
from signalmodel_fisp_epg_numpy import calculate_orthogonality, calculate_signal_fisp
#%%
beats = 15
shots = 47
n_ex = beats * shots
# fa = np.loadtxt('/home/tomgriesler/Documents/UM/code/abdominal/textfiles/FA_FISP_sydney.txt')
fa = np.loadtxt('/home/tomgr/Documents/abdominal/textfiles/FA_FISP_sydney.txt')[:n_ex]
tr = np.zeros(n_ex)
tr_offset = 5.4
# ph = np.zeros(n_ex)
ph = 3. * np.arange(n_ex).cumsum()

# # Jesse's cardiac T1T2 sequence
# prep = [1, 0, 2, 2] * 4
# ti = [21, 0, 0, 0, 100, 0, 0, 0, 250, 0, 0, 0, 400, 0, 0, 0]
# t2te = [0, 0, 40, 80] * 4

# Sydney's cardiac T1T2T1rho sequence
prep = [1, 0, 3, 3, 3] + [1, 0, 2, 2, 2] * 2
ti = [21, 0, 0, 0, 0] * 3
t2te = [0, 0, 30, 50, 60] + [0, 0, 30, 50, 80] * 2

te = 1.

# for ii in range(1, beats):
#     tr[ii*shots-1] += (total_dur - np.sum(ti) - np.sum(t2te) - beats*shots*tr_offset)*1e3/(beats-1)

for ii in range(beats):
    tr[(ii+1)*shots-1] += 1e6 - (ti[ii] + t2te[ii] + tr_offset*shots)*1e3
    
#%%
costfunction = 'crlb_orth'

# Relaxation times of healthy myocardium
t1 = [1000, 1100]
t2 = [44, 50]
m0 = 1
t1rho = [50, 90]

#%%
mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te)

#%%
visualize_sequence(mrf_seq, True)
plt.ylim(0)

#%%
mrf_seq.calc_signal_fisp(t1, t2, m0, t1rho=t1rho)

#%%
plt.plot(np.real(mrf_seq.signal))
plt.plot(np.imag(mrf_seq.signal))

# %%
mrf_seq.calc_cost(costfunction, t1, t2, m0, t1rho=t1rho)
print(mrf_seq.cost)

#%%
mrf_seq.calc_cost(costfunction, t1, t2, m0, t1rho=t1rho)

#%%
weightingmatrix = create_weightingmatrix('T1, T2, T1rho', t1, t2, t1rho, dims=4)
res = []
res_t1 = []
res_t2 = []
res_t1rho = []

for phase_inc in tqdm(np.arange(0, 10)):
    ph = phase_inc*np.arange(beats*shots).cumsum()
    mrf_seq = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te)
    mrf_seq.calc_cost(costfunction, t1, t2, m0, t1rho=t1rho)
    w_crlb = np.multiply(weightingmatrix, mrf_seq.cost)
    res.append(np.sum(w_crlb))
    res_t1.append(w_crlb[0])
    res_t2.append(w_crlb[1])
    res_t1rho.append(w_crlb[3])

#%%
plt.plot(res, label='cost$_{T1,T2}$:\t' + f'-{(1-min(res)/res[0])*100:.1f}%')
plt.plot(res_t1, label='cost$_{T1}$:\t' + f'-{(1-min(res_t1)/res_t1[0])*100:.1f}%')
plt.plot(res_t2, label='cost$_{T2}$:\t' + f'-{(1-min(res_t2)/res_t2[0])*100:.1f}%')
plt.plot(res_t1rho, label='cost$_{T1rho}$:\t' + f'-{(1-min(res_t1rho)/res_t1rho[0])*100:.1f}%')
plt.ylim(0)
plt.xlabel('$\Delta \Phi$ [deg]')
plt.ylabel('CRLB')
plt.legend()
plt.show()

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

signal_1 = calculate_signal_fisp(150, 20, 1, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)
signal_2 = calculate_signal_fisp(828, 72, 1, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)

plt.plot(-np.imag(signal_1))
plt.plot(-np.imag(signal_2))

# %%
calculate_orthogonality([150, 828], [20, 72], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, 1.)
# %%
