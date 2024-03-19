#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from signalmodel_fisp_epg_numpy import calculate_signal_fisp, calculate_signal_fisp_t1var

#%%
seqdir = Path('/scratch/abdominal/data/sequences/yun/textfiles_yun')

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

#%%
trange = np.linspace(0, sum(tr)*1e-3+ti, 100)
plt.plot(trange, [t1var(t) for t in trange])
plt.ylim(0)
plt.xlabel('t [ms]')
plt.ylabel('T1 [ms]')
#%%
signal_t1high = calculate_signal_fisp(1000, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1low = calculate_signal_fisp(100, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
signal_t1var = calculate_signal_fisp_t1var(t1var, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)

#%%
plt.plot(-np.imag(signal_t1high), label='$T_1=1000\,$ms')
plt.plot(-np.imag(signal_t1low))
plt.plot(-np.imag(signal_t1var))
plt.legend()

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
