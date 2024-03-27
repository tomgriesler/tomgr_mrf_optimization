#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from pathlib import Path
from numba import jit

#%%
from signalmodel_fisp_epg_numpy import calculate_signal_fisp_t1var, calculate_signal_fisp

#%% Parameters from Parker MRM 2006
A1 = 0.809*60       # [mmol*s]
A2 = 0.330*60       # [mmol*s]
T1 = 0.17046*60     # [s]
T2 = 0.365*60       # [s]
sigma1 = 0.0563*60  # [s]
sigma2 = 0.132*60   # [s]
alpha = 1.050       # [mmol]
beta = 0.1685/60    # [s^-1]
s = 38.078/60       # [s^-1]
tau = 0.483*60      # [s]
Hct = 0.42          
r = 4.5e-3          # [s^-1*(mmol/liter)^-1]

#%%
@jit(nopython=True)
def C_b(t):
    return A1/sigma1/np.sqrt(2*np.pi) * np.exp(-(t-T1)**2 / (2*sigma1**2))\
            + A2/sigma2/np.sqrt(2*np.pi) * np.exp(-(t-T2)**2 / (2*sigma2**2))\
            + alpha*np.exp(-beta*t) / (1+np.exp(-s*(t-tau)))


@jit(nopython=True)
def C_p(t):
    return C_b(t) / (1-Hct)


def C_t(t, k_trans, v_p, v_e):
    return v_p * C_p(t) + k_trans * integrate.quad(lambda x: (C_p(x) * np.exp(-k_trans * (t-x) / v_e)), 0, t)[0]


def t1(t, t10, r, k_trans, v_p, v_e):
    return 1/(1/t10 + r*C_t(t, k_trans, v_p, v_e))

#%%
t_arr = np.linspace(0, 60, 100)
plt.plot(t_arr, C_b(t_arr))

# %%
k_trans = 0.1
v_p = 0.02
v_e = 0.3

#%%
plt.plot(t_arr, [C_t(t, k_trans, v_p, v_e) for t in t_arr])

# %%
for k_trans in np.linspace(0.1, 0.4, 4, endpoint=True):
    plt.plot(t_arr, [t1(t, 1000, r, k_trans, v_p, v_e) for t in t_arr])

#%%
seqdir = Path('/home/tomgriesler/Documents/UM/code/abdominal/textfiles')
fa = np.loadtxt(seqdir/'FA_FISP.txt')
tr = np.loadtxt(seqdir/'TR_FISP.txt')
ti =  tr[0]
tr = tr[1:]*1e3
ph = 117*np.cumsum(np.arange(len(fa)))
t2 = 100

n = 4
fa = np.tile(fa, n)
tr = np.tile(tr, n)
ph = np.tile(ph, n)

prep = [1]
ti = [20]
t2te = [0]

t10 = 1000

#%%
plt.plot(fa)

#%%
signal_t1var, t_arr = calculate_signal_fisp_t1var(lambda x: t1(x, t10, r, k_trans, v_p, v_e), t2, 1, 1, len(fa), fa, tr, ph, prep, ti, t2te, 0, 1, 1, return_t_arr=True)

signal_t1const_high = calculate_signal_fisp_t1var(lambda x: 1000, t2, 1, 1, len(fa), fa, tr, ph, prep, ti, t2te, 0, 1, 1)

signal_t1const_low = calculate_signal_fisp_t1var(lambda x: 100, t2, 1, 1, len(fa), fa, tr, ph, prep, ti, t2te, 0, 1, 1)

# %%
plt.plot(t_arr, -np.imag(signal_t1var))
plt.plot(t_arr, -np.imag(signal_t1const_high))
plt.plot(t_arr, -np.imag(signal_t1const_low))

# %%
k_trans_arr = np.linspace(0.1, 0.5, 5, endpoint=True)
v_p_arr = np.linspace(0.02, 0.1, 5, endpoint=True)
v_e_arr = np.linspace(0.1, 0.7, 7, endpoint=True)
t10_arr = np.linspace(500, 1500, 11, endpoint=True)

#%% Test T2 dependency of FLASH sequence
for t2 in np.linspace(10, 200, 11, endpoint=True):
    plt.plot(-np.imag(calculate_signal_fisp(1000, t2, 1, 1, len(fa), fa, tr, ph, prep, ti, t2te, 0, 1, 1)))


#%%
dictsize = len(k_trans_arr) * len(v_p_arr) * len(v_e_arr) * len(t10_arr)
dict_signals = np.empty((dictsize, len(fa)))
dict_params = np.empty((dictsize, 4))

# %%
count = 0
for k_trans in k_trans_arr: 
    for v_p in v_p_arr:
        for v_e in v_e_arr:
            for t10 in t10_arr:
                dict_signals[count] = -np.imag(calculate_signal_fisp_t1var(lambda x: t1(x, t10, r, k_trans, v_p, v_e), t2, 1, 1, len(fa), fa, tr, ph, prep, ti, t2te, 0, 1, 1))
                dict_params[count] = [k_trans, v_p, v_e, t10]
                count += 1
                print(count, end='\r')
# %%
