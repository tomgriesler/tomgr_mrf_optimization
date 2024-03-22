#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def aif(t, A, B, C):
    return A * t**B * np.exp(-t/C)

#%%
t = np.linspace(0, 40, 100)

#%%
plt.plot(t, aif(t, 1, 1, 1))
# %%
for B in range(1, 5):
    plt.plot(t, aif(t, 1, B, 1))
# %%
for C in range(1, 5):
    plt.plot(t, aif(t/2, 1, 1, C))
# %%
