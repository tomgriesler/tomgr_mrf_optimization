#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#%%
A1 = 0.809
A2 = 0.330
T1 = 0.17046
T2 = 0.365
sigma1 = 0.0563
sigma2 = 0.132
alpha = 1.050
beta = 0.1685
s = 38.078
tau = 0.483
Hct = 0.42

#%%
def C_b(t):
    return A1/sigma1/np.sqrt(2*np.pi) * np.exp(-(t-T1)**2 / (2*sigma1**2))\
            + A2/sigma2/np.sqrt(2*np.pi) * np.exp(-(t-T2)**2 / (2*sigma2**2))\
            + alpha*np.exp(-beta*t) / (1+np.exp(-s*(t-tau)))
    

def C_p(t):
    return C_b(t) / (1-Hct)

def C_t(t, k_trans, v_p, v_e):
    return v_p * C_p(t) + k_trans * integrate.quad(lambda x: (C_p(x) * np.exp(-k_trans * (t-x) / v_e)), 0, t)[0]

#%%
t = np.linspace(0, 2, 100)
plt.plot(t, C_b(t))

# %%
k_trans = 0.3
v_p = 0.02
v_e = 0.5

plt.plot(t, [C_t(ti, k_trans, v_p, v_e) for ti in t])
# %%
