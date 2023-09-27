#%%
import json
import matplotlib.pyplot as plt
import numpy as np
# %%
with open('/home/tomgr/Documents/code/abdominal/results_optim/230926_1709_adv2.json', 'r') as handle: 
    prot = json.load(handle)

#%%
#%%
blocks = {
    'noPrep':
    {
        'fa': [],
        'tr': []
    },
    'TI12': {
        'fa': [180],
        'tr': [12]
    },
    'TI300':
    {
        'fa': [180],
        'tr': [300]
    },
    'T2prep40':
    {
        'fa': [90, -90], 
        'tr': [40, 20]
    },
    'T2prep80':
    {
        'fa': [90, -90],
        'tr': [80, 20]
    },
    'T2prep160':
    {
        'fa': [90, -90],
        'tr': [160, 20]
    }
}

# %%
fa = prot['sequences']['fa_ref']
tr = prot['sequences']['tr_ref']
tr_jaubert = np.load('/home/tomgr/Documents/code/abdominal/tr_jaubert.npy')

sequence = prot['sequence_ref']
# waittimes = prot['waittimes_ref']
waittimes = [508, 500, 460, 420, 340, 220, 500, 460, 420, 340, 508, 500]
preptime = 0
preptimes = [preptime]
for i in range(len(sequence)-1):
    preptime += sum(blocks[sequence[i]]['tr']) + sum(tr_jaubert[:-1]) + waittimes[i]
    preptimes.append(preptime)

# %%
plt.plot([sum(tr[:i]) for i in range(len(tr))], fa, '.', color='black')
plt.grid()
plt.xlabel('Time [ms]')
plt.ylabel('FA [deg]')

colormap = {prep: color for prep, color in zip(list(set(sequence)), plt.rcParams['axes.prop_cycle'].by_key()['color'])}

for i in range(12): 
    plt.axvspan(preptimes[i], preptimes[i]+sum(blocks[sequence[i]]['tr']), color=colormap[sequence[i]], label=sequence[i], alpha=0.5)
    plt.axvspan(preptimes[i]+sum(blocks[sequence[i]]['tr']), preptimes[i]+sum(blocks[sequence[i]]['tr'])+sum(tr_jaubert[:-1]), color='gray', alpha=0.5, label='acquisition')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', ncols=3)

plt.ylim(0, 20)

plt.tight_layout()
# %%
