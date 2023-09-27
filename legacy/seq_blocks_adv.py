#%%
import numpy as np
import random
import json
from datetime import datetime
# %%
import sys
sys.path.append('/home/tomgr/Documents/tg_mrf_optimization')

#%%
from costfunctions import calculate_crlb_sc_epg

# %%
def divide_into_random_integers(N, n):

    positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
    integers = [positions[i+1]-positions[i] for i in range(n)]

    return integers

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
T1 = 661.5
T2 = 56.8
M0 = 1
TE = 1.4

#%%
ref_sequence = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']
ref_waittimes = [508, 500, 460, 420, 340, 220, 500, 460, 420, 340, 508, 500]

fa_ref = np.concatenate([np.concatenate([blocks[name]['fa'], np.full(35, 15)]) for name in ref_sequence])
tr_ref = np.concatenate([np.concatenate([blocks[name]['tr'], np.full(34, 20), [wait_time]]) for name, wait_time in zip(ref_sequence, ref_waittimes)])

crlb_ref = calculate_crlb_sc_epg(T1, T2, M0, fa_ref, tr_ref, TE, [], []).item()

#%%
first = True
count = 0
preparations = ['noPrep']*3 + ['TI12']*2 + ['TI300'] + ['T2prep40']*2 + ['T2prep80']*2 + ['T2prep160']*2

t0 = datetime.now()

try:
    while True:

        newlist = preparations.copy()
        random.shuffle(newlist)

        waittimes = divide_into_random_integers(sum(ref_waittimes), 12)

        fa = np.concatenate([np.concatenate([blocks[name]['fa'], np.full(35, 15)]) for name in newlist])
        tr = np.concatenate([np.concatenate([blocks[name]['tr'], np.full(34, 20), [wait_time]]) for name, wait_time in zip(newlist, waittimes)])

        crlb = calculate_crlb_sc_epg(T1, T2, M0, fa, tr, TE, [], []).item()

        if first:
            crlb_min = crlb
            crlb_max = crlb
            sequence_min = newlist
            sequence_max = newlist
            waittimes_min = waittimes
            waittimes_max = waittimes
            count_min = count
            count_max = count
            first = False
        
        else:
            if crlb < crlb_min:
                crlb_min = crlb
                sequence_min = newlist
                waittimes_min = waittimes
                count_min = count

            elif crlb > crlb_max:
                crlb_max = crlb
                sequence_max = newlist
                waittimes_max = waittimes
                count_max = count

        count += 1

        print(f'{count} iters. Min CRLB: {crlb_min:.3f} at iter {count_min}. Impr of {(1-crlb_min/crlb_ref)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')

except KeyboardInterrupt:

    duration = str(datetime.now()-t0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')[2:]

    print(f'\n\nMin CRLB: {crlb_min:.3f} at iteration {count_min}.')
    print(f'Max CRLB: {crlb_max:.3f} at iteration {count_max}.')
    print(f'Ref CRLB: {crlb_ref:.3f}.')
    print(f'Improvement of {(1-crlb_min/crlb_ref)*100:.2f}%.')
    print(f'Time elapsed: {duration}.')

fa_min = list(np.concatenate([np.concatenate([blocks[name]['fa'], np.full(35, 15)]) for name in sequence_min]))
tr_min = list(np.concatenate([np.concatenate([blocks[name]['tr'], np.full(34, 20), [wait_time]]) for name, wait_time in zip(sequence_min, waittimes_min)]))

fa_max = list(np.concatenate([np.concatenate([blocks[name]['fa'], np.full(35, 15)]) for name in sequence_max]))
tr_max = list(np.concatenate([np.concatenate([blocks[name]['tr'], np.full(34, 20), [wait_time]]) for name, wait_time in zip(sequence_max, waittimes_max)]))

fa_ref = list(np.concatenate([np.concatenate([blocks[name]['fa'], np.full(35, 15)]) for name in ref_sequence]))
tr_ref = list(np.concatenate([np.concatenate([blocks[name]['tr'], np.full(34, 20), [wait_time]]) for name, wait_time in zip(ref_sequence, ref_waittimes)]))

prot = {
    'crlb_min': crlb_min,
    'crlb_max': crlb_max,
    'sequence_min': sequence_min,
    'waittimes_min': waittimes_min,
    'sequence_max': sequence_max,
    'waittimes_max': waittimes_max,
    'count': count,
    'count_min': count_min,
    'count_max': count_max,
    'duration': duration,
    'T1': T1,
    'T2': T2, 
    'M0': M0, 
    'TE': TE, 
    'crlb_ref': crlb_ref,
    'sequence_ref': ref_sequence,
    'sequences':{
        'fa_min': fa_min,
        'tr_min': tr_min,
        'fa_max': fa_max,
        'tr_max': tr_max,
        'fa_ref': fa_ref,
        'tr_ref': tr_ref
    }
}

#%%
with open(f'/home/tomgr/Documents/code/abdominal/{timestamp}_adv.json', 'w') as handle:
    json.dump(prot, handle, indent='\t')
# %%
