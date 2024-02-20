import numpy as np
import random
from datetime import datetime

from utils.abdominal_tools import BLOCKS, divide_into_random_floats, MRFSequence


def randomized_sequences(costfunction, target_t1, target_t2, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights=None, min_beats=1, n_iter_max=np.inf, inv_eff=0.95, delta_B1=1., phase_inc=0.):

    sequences = []

    count = 0

    max_prep_dur = max([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_modules])

    max_beats = total_dur // (max_prep_dur+shots*const_tr)

    if min_beats is None:
        min_beats = max_beats

    print(f'Total sequence duration: {total_dur:.0f} ms.\nMax num of preps: {max_beats:.0f}.')

    t0 = datetime.now()

    try:
        while True:

            beats = random.randint(min_beats, max_beats)
            n_ex = beats*shots

            prep_order = random.choices(prep_modules, weights=prep_module_weights, k=beats)
            prep = [BLOCKS[name]['prep'] for name in prep_order]
            ti = [BLOCKS[name]['ti'] for name in prep_order]
            t2te = [BLOCKS[name]['t2te'] for name in prep_order]

            prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_order])

            waittime_tot = int(total_dur - beats*shots*const_tr - prep_time_tot)
            waittimes = divide_into_random_floats(waittime_tot, beats-1)

            fa = np.repeat(random.choices((const_fa), k=beats), shots) if type(const_fa) == list else np.full(beats*shots, const_fa)

            # Ensure that not all flipangles are zero
            if np.count_nonzero(fa) == 0:
                continue

            tr = np.full(beats*shots, 0)

            for ii in range(len(waittimes)):
                tr[(ii+1)*shots-1] += waittimes[ii]*1e3

            ph = phase_inc*np.arange(n_ex).cumsum()

            mrf_sequence = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, const_tr, te)

            mrf_sequence.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)

            sequences.append(mrf_sequence)

            count += 1

            timediff = datetime.now()-t0

            print(f'{count} iters. Time: {str(timediff).split(".")[0]}. {count/timediff.total_seconds():.2f} its/s.', end='\r')

            if count >= n_iter_max:
                break

    except KeyboardInterrupt:
        pass
        
    return sequences