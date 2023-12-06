import numpy as np
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from abdominal_tools import BLOCKS, divide_into_random_integers, MRFSequence

def optimize_sequence_worker(seed, min_num_preps, max_num_preps, total_dur, prep_modules, prep_module_weights, costfunction, target_t1, target_t2, target_m0, shots, const_fa, const_tr, te, inv_eff, delta_B1, phase_inc, n_iter_max):
    
    random.seed(seed)
    
    sequences = []
    count = 0

    while True:
        beats = random.randint(min_num_preps, max_num_preps)
        n_ex = beats * shots

        prep_order = random.choices(prep_modules, weights=prep_module_weights, k=beats)
        prep = [BLOCKS[name]['prep'] for name in prep_order]
        ti = [BLOCKS[name]['ti'] for name in prep_order]
        t2te = [BLOCKS[name]['t2te'] for name in prep_order]

        prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_order])

        waittime_tot = int(total_dur - beats * shots * const_tr - prep_time_tot)

        waittimes = divide_into_random_integers(waittime_tot, beats - 1)

        fa = np.repeat(random.choices((const_fa), k=beats), shots) if type(const_fa) == list else np.full(beats * shots, const_fa)

        tr = np.full(beats * shots, 0)

        for ii in range(len(waittimes)):
            tr[(ii + 1) * shots - 1] += waittimes[ii] * 1e3

        ph = phase_inc * np.arange(n_ex).cumsum()

        mrf_sequence = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, const_tr, te)

        mrf_sequence.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)

        sequences.append(mrf_sequence)

        count += 1

        if count >= n_iter_max:
            break

    return sequences

def optimize_sequence(costfunction, target_t1, target_t2, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights=None, min_num_preps=1, n_iter_max=np.inf, inv_eff=0.95, delta_B1=1., phase_inc=0., num_workers=8):
    sequences = []

    max_prep_dur = max([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_modules])

    max_num_preps = total_dur // (max_prep_dur + shots * const_tr)

    print(f'Total sequence duration: {total_dur:.0f} ms.\nMax num of preps: {max_num_preps:.0f}.')

    t0 = datetime.now()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for seed in range(num_workers):
            futures.append(executor.submit(optimize_sequence_worker, seed, min_num_preps, max_num_preps, total_dur, prep_modules, prep_module_weights, costfunction, target_t1, target_t2, target_m0, shots, const_fa, const_tr, te, inv_eff, delta_B1, phase_inc, n_iter_max/num_workers))
        
        for future in as_completed(futures):
            sequences.extend(future.result())

    timediff = datetime.now() - t0
    print(f'Total time: {str(timediff).split(".")[0]}. {len(sequences) / timediff.total_seconds():.2f} its/s.')

    return sequences