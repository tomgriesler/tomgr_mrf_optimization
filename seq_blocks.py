#%%
import numpy as np
import random
import json
from datetime import datetime
# %%
import sys
sys.path.append('/home/tomgr/Documents/tg_mrf_optimization')
from costfunctions import calculate_crlb_sc_epg
# %%
blocks = {
    'noPrep':
    {
        'fa': np.full(35, 15),
        'tr': np.concatenate([np.full(34, 20), [500]])
    },
    'TI12': {
        'fa': np.concatenate([[180], np.full(35, 15)]),
        'tr': np.concatenate([[12], np.full(34, 20), [508]])
    },
    'TI300':
    {
        'fa': np.concatenate([[180], np.full(35, 15)]),
        'tr': np.concatenate([[300], np.full(34, 20), [220]])
    },
    'T2prep40':
    {
        'fa': np.concatenate([[90, -90], np.full(35, 15)]),
        'tr': np.concatenate([[40], np.full(35, 20), [460]])
    },
    'T2prep80':
    {
        'fa': np.concatenate([[90, -90], np.full(35, 15)]),
        'tr': np.concatenate([[80], np.full(35, 20), [420]])
    },
    'T2prep160':
    {
        'fa': np.concatenate([[90, -90], np.full(35, 15)]),
        'tr': np.concatenate([[160], np.full(35, 20), [340]])
    }
}
# %%
T1 = 661.5
T2 = 56.8
M0 = 1
TE = 1.4

#%%
ref_sequence = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']

fa_ref = np.concatenate([blocks[name]['fa'] for name in ref_sequence])
tr_ref = np.concatenate([blocks[name]['tr'] for name in ref_sequence])

crlb_ref = calculate_crlb_sc_epg(T1, T2, M0, fa_ref, tr_ref, TE, [], []).item()

#%%
first = True
count = 0
blocklist = ['noPrep']*3 + ['TI12']*2 + ['TI300'] + ['T2prep40']*2 + ['T2prep80']*2 + ['T2prep160']*2

t0 = datetime.now()

try:
    while True:

        newlist = blocklist.copy()
        random.shuffle(newlist)

        fa = np.concatenate([blocks[name]['fa'] for name in newlist])
        tr = np.concatenate([blocks[name]['tr'] for name in newlist])

        crlb = calculate_crlb_sc_epg(T1, T2, M0, fa, tr, TE, [], []).item()

        if first:
            crlb_min = crlb
            crlb_max = crlb
            sequence_min = newlist
            sequence_max = newlist
            count_min = count
            count_max = count
            first = False
        
        else:
            if crlb < crlb_min:
                crlb_min = crlb
                sequence_min = newlist
                count_min = count

            elif crlb > crlb_max:
                crlb_max = crlb
                sequence_max = newlist
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


prot = {
    'crlb_min': crlb_min,
    'crlb_max': crlb_max,
    'sequence_min': sequence_min,
    'sequence_max': sequence_max,
    'count': count,
    'count_min': count_min,
    'count_max': count_max,
    'duration': duration,
    'T1': T1,
    'T2': T2, 
    'M0': M0, 
    'TE': TE, 
    'crlb_ref': crlb_ref,
    'sequence_ref': ref_sequence
}

#%%
with open(f'/home/tomgr/Documents/code/abdominal/{timestamp}.json', 'w') as handle:
    json.dump(prot, handle, indent='\t')
# %%
