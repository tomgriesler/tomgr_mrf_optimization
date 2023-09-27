#%%
import numpy as np
import random
import json
from datetime import datetime
import pickle

from abdominal_tools import RESULTSPATH, BLOCKS, divide_into_random_integers, MRFSequence

# %%
T1 = 661.5
T2 = 56.8
M0 = 1
TE = 1.4

#%%
fa_jaubert = np.load('/home/tomgr/Documents/code/abdominal/fa_jaubert.npy')
tr_jaubert = np.load('/home/tomgr/Documents/code/abdominal/tr_jaubert.npy')
prep_modules = ['noPrep', 'TI12', 'TI300', 'T2prep40', 'T2prep80', 'T2prep160']

#%% Reference sequence
prep_order_ref = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']
waittimes_ref = [508, 500, 460, 420, 340, 220, 500, 460, 420, 340, 508, 500]

mrf_sequence_ref = MRFSequence(prep_order_ref, waittimes_ref, fa_jaubert, tr_jaubert, BLOCKS)
mrf_sequence_ref.calc_crlb(T1, T2, M0, TE)

#%%
track_crlbs = True

crlbs, crlbs_T1, crlbs_T2, crlbs_M0 = [], [], [], []

best_sequences, worst_sequences = [], []

count, count_min, count_max = 0, 0, 0

mrf_sequence_best = mrf_sequence_ref
mrf_sequence_worst = mrf_sequence_ref

t0 = datetime.now()
timestamp = t0.strftime('%Y%m%d_%H%M')[2:]

try:
    while True:

        prep_order = random.choices(prep_modules, k=12)

        prep_time_tot = sum(np.concatenate([BLOCKS[module]['tr'] for module in prep_order]))

        waittime_tot = int(12*1.2e3 - 12*sum(tr_jaubert[:-1]) - prep_time_tot)

        waittimes = divide_into_random_integers(waittime_tot, 12)

        mrf_sequence = MRFSequence(prep_order, waittimes, fa_jaubert, tr_jaubert, BLOCKS)

        mrf_sequence.calc_crlb(T1, T2, M0, TE)

        if track_crlbs:
            crlbs.append(mrf_sequence.crlb)    
            crlbs_T1.append(mrf_sequence.crlb_T1)    
            crlbs_T2.append(mrf_sequence.crlb_T2)    
            crlbs_M0.append(mrf_sequence.crlb_M0)    
        

        if len(best_sequences) < 20:
            best_sequences.append(mrf_sequence)
            best_sequences.sort(key=lambda x: x.crlb)
            worst_sequences.append(mrf_sequence)
            worst_sequences.sort(key=lambda x: x.crlb)

        elif mrf_sequence.crlb < best_sequences[-1].crlb:
            best_sequences[-1] = mrf_sequence
            best_sequences.sort(key=lambda x: x.crlb)

        elif mrf_sequence.crlb > worst_sequences[0].crlb:
            worst_sequences[0] = mrf_sequence
            worst_sequences.sort(key=lambda x: x.crlb)


        if mrf_sequence.crlb < mrf_sequence_best.crlb:
            mrf_sequence_best = mrf_sequence
            count_min = count

        elif mrf_sequence.crlb > mrf_sequence_worst.crlb:
            mrf_sequence_worst = mrf_sequence
            count_max = count

        count += 1

        print(f'{count} iters. Min CRLB: {mrf_sequence_best.crlb:.3f} at iter {count_min}. Impr of {(1-mrf_sequence_best.crlb/mrf_sequence_ref.crlb)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')

except KeyboardInterrupt:

    duration = str(datetime.now()-t0)

    print(f'\n\nMin CRLB: {mrf_sequence_best.crlb:.3f} at iteration {count_min}.')
    print(f'Max CRLB: {mrf_sequence_worst.crlb:.3f} at iteration {count_max}.')
    print(f'Ref CRLB: {mrf_sequence_ref.crlb:.3f}.')
    print(f'Improvement of {(1-mrf_sequence_best.crlb/mrf_sequence_ref.crlb)*100:.2f}%.')
    print(f'Time elapsed: {duration}.')

    [crlbs, crlbs_T1, crlbs_T2, crlbs_M0] = np.transpose(sorted(zip(crlbs, crlbs_T1, crlbs_T2, crlbs_M0)))

#%%
resultspath = RESULTSPATH / timestamp
resultspath.mkdir()

with open(resultspath/'mrf_sequence_best.pkl', 'wb') as handle:
    pickle.dump(mrf_sequence_best, handle)

with open(resultspath/'mrf_sequence_worst.pkl', 'wb') as handle:
    pickle.dump(mrf_sequence_worst, handle)

with open(resultspath/'mrf_sequence_ref.pkl', 'wb') as handle:
    pickle.dump(mrf_sequence_ref, handle)

prot = {
    'count': count,
    'crlb_min': mrf_sequence_best.crlb,
    'reduction': 1-mrf_sequence_best.crlb/mrf_sequence_ref.crlb,
    'count_min': count_min,
    'count_max': count_max,
    'duration': duration,
    'T1': T1,
    'T2': T2, 
    'M0': M0, 
    'TE': TE, 
    'crlbs': crlbs, 
    'crlbs_T1': crlbs_T1,
    'crlbs_T2': crlbs_T2,
    'crlbs_M0': crlbs_M0
}
with open(resultspath/'prot.json', 'w') as handle:
    json.dump(prot, handle, indent='\t')

with open(resultspath/'best_sequences.pkl', 'wb') as handle:
    pickle.dump({i: best_sequences[i] for i in range(len(best_sequences))}, handle)
with open(resultspath/'worst_sequences.pkl', 'wb') as handle:
    pickle.dump({i: best_sequences[i] for i in range(len(worst_sequences))}, handle)

# %%
