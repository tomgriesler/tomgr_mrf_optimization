#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from signalmodel_fisp_epg_numpy import calculate_signal_fisp, calculate_signal_fisp_t1var

#%%
seqdir = Path('/scratch/abdominal/data/sequences/yun/textfiles_yun')

#%%
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %%
fa = np.loadtxt(seqdir/'FA_FISP.txt')
tr = np.loadtxt(seqdir/'TR_FISP.txt')
ti =  tr[0]
tr = tr[1:]*1e3
ph = 117*np.cumsum(np.arange(len(fa)))
seqdur = sum(tr)*1e-3 + ti

#%%
fa = np.tile(fa, 3)
tr = np.tile(tr, 3)
ph = np.tile(ph, 3)
seqdur *= 3

# %%
t2 = 100
m0 = 1
beats = 1
shots = len(fa)

# %%
def t1var(t):
    return 1000-900/seqdur*t

def t1var2(t):
    return 1000-1800/seqdur*t if t<seqdur/2 else -800+1800/seqdur*t

def t1var3(t):
    return 1000-1800/seqdur*t if t<seqdur/2 else 100

def t1var4(t):
    return 1000-450/seqdur*t

def t1var5(t):
    return 1000-900/seqdur*t if t<seqdur/2 else 550

#%%
trange = np.linspace(0, sum(tr)*1e-3+ti, 100)
# plt.plot(trange, np.full_like(trange, 1000), label='$T_1=1000\,$ms', ls='--', color=colors[0])
# plt.plot(trange, np.full_like(trange, 100), label='$T_1=100\,$ms', ls='--', color=colors[1])
plt.plot(trange, [t1var(t) for t in trange], label='$T_1=T_1^A(t)$', color=colors[2])
# plt.plot(trange, [t1var2(t) for t in trange], label='$T_1=T_1^B(t)$', color=colors[3])
plt.plot(trange, [t1var3(t) for t in trange], label='$T_1=T_1^C(t)$', color=colors[4])
plt.plot(trange, [t1var4(t) for t in trange], label='$T_1=T_1^D(t)$', color=colors[5])
plt.plot(trange, [t1var5(t) for t in trange], label='$T_1=T_1^E(t)$', color=colors[6])
plt.ylim(0)
plt.xlabel('t [ms]')
plt.ylabel('T1 [ms]')
plt.legend(ncols=2, loc='upper center')
#%%
signal_t1high = calculate_signal_fisp(1000, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1low = calculate_signal_fisp(100, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var = calculate_signal_fisp_t1var(t1var, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var2 = calculate_signal_fisp_t1var(t1var2, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var3 = calculate_signal_fisp_t1var(t1var3, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var4 = calculate_signal_fisp_t1var(t1var4, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var5 = calculate_signal_fisp_t1var(t1var5, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)

#%%
# plt.plot(-np.imag(signal_t1high), label='$T_1=1000\,$ms', ls='--', color=colors[0])
# plt.plot(-np.imag(signal_t1low), label='$T_1=100\,$ms', ls='--', color=colors[1])
plt.plot(-np.imag(signal_t1var), label='$T_1=T_1^A(t)$', color=colors[2])
# plt.plot(-np.imag(signal_t1var2), label='$T_1=T_1^B(t)$', color=colors[3])
plt.plot(-np.imag(signal_t1var3), label='$T_1=T_1^C(t)$', color=colors[4])
plt.plot(-np.imag(signal_t1var4), label='$T_1=T_1^D(t)$', color=colors[5])
plt.plot(-np.imag(signal_t1var5), label='$T_1=T_1^E(t)$', color=colors[6])
plt.legend(ncol=2)

# %%
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(fa)
plt.gca().set_xticklabels([])
plt.ylabel('FA [deg]')

plt.subplot(2, 1, 2)
plt.plot(tr)
plt.xlabel('TR Index')
plt.ylabel('TR [ms]')
# %%
