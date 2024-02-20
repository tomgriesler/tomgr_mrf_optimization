#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.abdominal_tools import MRFSequence, create_weightingmatrix
from utils.visualization import visualize_sequence

#%%
t1 = 660.
t2 = 40.
t1rho = None
m0 = 1. 

weightingmatrix = create_weightingmatrix('T1, T2', t1, t2, t1rho, dims=3)

#%%
# seqpaths = [
#     '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc0deg/textfiles_hamilton_10s_16_phinc0deg',
#     '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc3deg/textfiles_hamilton_10s_16_phinc3deg',
#     '/scratch/abdominal/data/sequences/231211_171934/231211_171934_best_T1T2/textfiles_231211_171934_best_T1T2',
#     '/scratch/abdominal/data/sequences/231211_171937/231211_171937_best_T1T2/textfiles_231211_171937_best_T1T2'
# ]

seqpaths = [
    '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc0deg/textfiles_hamilton_10s_16_phinc0deg',
    '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc3deg/textfiles_hamilton_10s_16_phinc3deg',
    '/scratch/abdominal/data/sequences/hamilton_mod_10s_31_phinc0deg/textfiles_hamilton_mod_10s_31_phinc0deg',
    '/scratch/abdominal/data/sequences/hamilton_mod_10s_31_phinc3deg/textfiles_hamilton_mod_10s_31_phinc3deg'
]

seqpaths = [Path(seqpath) for seqpath in seqpaths]

#%%
mrf_seqs = []

for seqpath in seqpaths:

    fa = np.loadtxt(seqpath/'FA_FISP.txt')
    tr = np.loadtxt(seqpath/'TR_FISP.txt')
    ph = np.loadtxt(seqpath/'PH_FISP.txt')
    prep = np.loadtxt(seqpath/'PREP_FISP.txt')
    ti = np.loadtxt(seqpath/'TI_FISP.txt')
    t2te = np.multiply((prep==2)|(prep==3), np.loadtxt(seqpath/'T2TE_FISP.txt'))

    mrf_seq = MRFSequence(len(prep), int(len(fa)/len(prep)), fa, tr, ph, prep, ti, t2te, 5.4, 1.)
    mrf_seq.calc_cost('crlb', t1, t2, 1., t1rho=t1rho)

    mrf_seqs.append(mrf_seq)
    
# %%
for mrf_seq in mrf_seqs:
    print(np.multiply(weightingmatrix, mrf_seq.cost))

# %%
fig = plt.figure()

for ii in range(len(mrf_seqs)):
    plt.subplot(len(mrf_seqs), 2, 2*ii+1)
    visualize_sequence(mrf_seqs[ii], True)
    plt.ylim(0, 30)
    if ii==0:
        plt.title('Preparations & Flip Angles')
    if ii==3:
        plt.xlabel('Time in ms')
    else:
        plt.gca().set_xticklabels([])

    plt.subplot(len(mrf_seqs), 2, 2*ii+2)
    plt.plot(mrf_seqs[ii].ph)
    plt.yticks([])
    if ii==0:
        plt.title('Phase')
    if ii==3:
        plt.xlabel('TR Index')
    
plt.tight_layout()
# plt.savefig('/home/tomgr/Documents/temp/seqs_t1rho.png', dpi=300)
plt.show()
# %%
