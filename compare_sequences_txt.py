#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from abdominal_tools import MRFSequence, create_weightingmatrix, visualize_sequence

#%%
t1 = 1000.
t2 = 44.
t1rho = 50.
m0 = 1. 

weightingmatrix = create_weightingmatrix('T1, T2, T1rho', t1, t2, t1rho, dims=4)

#%%
# seqpaths = [
#     '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc0deg/textfiles_hamilton_10s_16_phinc0deg',
#     '/scratch/abdominal/data/sequences/hamilton_10s_16_phinc3deg/textfiles_hamilton_10s_16_phinc3deg',
#     '/scratch/abdominal/data/sequences/231211_171934/231211_171934_best_T1T2/textfiles_231211_171934_best_T1T2',
#     '/scratch/abdominal/data/sequences/231211_171937/231211_171937_best_T1T2/textfiles_231211_171937_best_T1T2'
# ]

seqpaths = [
    '/scratch/abdominal/data/sequences/sydney_15s_15_phinc0deg/textfiles_sydney_15s_15_phinc0deg',
    '/scratch/abdominal/data/sequences/sydney_15s_15_phinc3deg/textfiles_sydney_15s_15_phinc3deg',
    '/scratch/abdominal/data/sequences/231218_074414/231218_074414_best_T1T2T1rho/textfiles_231218_074414_best_T1T2T1rho',
    '/scratch/abdominal/data/sequences/231218_074418/231218_074418_best_T1T2T1rho/textfiles_231218_074418_best_T1T2T1rho'
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
    t2te = np.multiply(prep==2, np.loadtxt(seqpath/'T2TE_FISP.txt'))
    tsl = np.multiply(prep==3, np.loadtxt(seqpath/'T2TE_FISP.txt'))

    mrf_seq = MRFSequence(len(prep), int(len(fa)/len(prep)), fa, tr, ph, prep, ti, t2te, 5.4, 1., tsl)
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
plt.savefig('/home/tomgr/Documents/temp/seqs_t1rho.png', dpi=300)
plt.show()
# %%
