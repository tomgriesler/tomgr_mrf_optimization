#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from signalmodel_bssfp_flash_numpy import calculate_signal_flash, calculate_signal_bssfp
from signalmodel_fisp_epg_numpy import calculate_signal_fisp
from utils.abdominal_tools import MRFSequence

#%%
seqdir = Path('/scratch/abdominal/data/sequences/yun/textfiles_yun')

# %%
fa = np.loadtxt(seqdir/'FA_FISP.txt')
tr = np.loadtxt(seqdir/'TR_FISP.txt')*1e3
ti =  tr[0]
tr = tr[1:]

# %%
t1 = 1000
t2 = 100
m0 = 1
beats = 1
shots = len(fa)

# %%
ph = 117*np.cumsum(np.arange(len(fa)))
signal = calculate_signal_fisp(t1, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1, 1)
plt.plot(-np.imag(signal))
plt.plot(np.real(signal))

#%%
signal_flash = calculate_signal_flash(t1, t2, 1, 1, len(fa), fa, tr, np.zeros_like(fa), [1], [ti], [0], 0, 1, 1)

plt.figure()
plt.plot(-np.imag(signal_flash))
plt.plot(-np.imag(signal))

plt.figure()
plt.plot(np.real(signal_flash))
plt.plot(np.real(signal))

#%%
mrfseq = MRFSequence(1, len(fa), fa, np.zeros_like(fa), np.zeros_like(fa), [1], [20], [0], 12, 1)
plt.plot(-np.imag(mrfseq.calc_signal_fisp(t1, t2, m0, 1, return_result=True)))

# %%
n = 100
t1 = 600
t2 = 100
tr = np.full(n, 10)*1e3
te = 2
fa = np.full(n, 30)
ph = 117*np.cumsum(np.arange(n))
# ph = np.zeros(n)

signal = calculate_signal_fisp(t1, t2, 1., 1, n, fa, tr, ph, [0], [0], [0], 0, te)

plt.plot(np.real(signal))
plt.show()
plt.plot(np.imag(signal))

plt.plot(np.angle(signal))
plt.show()
# %%
ph = 1*np.cumsum(np.arange(len(fa)))
# ph = np.zeros_like(fa)
plt.plot(np.abs(calculate_signal_fisp(t1, t2, 1, 1, len(fa), fa, tr, ph, [1], [ti], [0], 0, 1)))
# %%
