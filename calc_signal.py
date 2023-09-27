#%%
import json
import matplotlib.pyplot as plt
import numpy as np

# %%
import sys
sys.path.append('/home/tomgr/Documents/tg_mrf_optimization')

#%%
from costfunctions import calculate_crlb_sc_epg
from signalmodel_epg import calculate_complex_signal_epg
# %%
with open('/home/tomgr/Documents/code/abdominal/230926_1255_adv.json', 'r') as handle:
    prot = json.load(handle)

# %%
fa = prot['sequences']['fa_min']
tr = prot['sequences']['tr_min']

# %%
T1 = 661.5
T2 = 56.8
M0 = 1
TE = 1.4

#%%
calculate_crlb_sc_epg(T1, T2, M0, fa, tr, TE, [], [])
# %%
signal = np.imag(calculate_complex_signal_epg(T1, T2, M0, fa, tr, TE, [], []).detach().numpy())
# %%
plt.plot(signal)
# %%
