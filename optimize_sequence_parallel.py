import numpy as np
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from abdominal_tools import BLOCKS, divide_into_random_floats, MRFSequence


def create_sequence(args):

    costfunction, target_t1, target_t2, target_t1rho, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights, min_beats, max_beats, inv_eff, delta_B1, phase_inc, optimize_positions = args

    beats = random.randint(min_beats, max_beats)
    n_ex = beats * shots

    prep_order = random.choices(prep_modules, weights=prep_module_weights, k=beats)
    prep = [BLOCKS[name]['prep'] for name in prep_order]
    ti = [BLOCKS[name]['ti'] for name in prep_order]
    t2te = [BLOCKS[name]['t2te'] for name in prep_order]
    tsl = [BLOCKS[name]['tsl'] for name in prep_order]

    prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] + BLOCKS[name]['tsl'] for name in prep_order])

    if optimize_positions:
        waittime_tot = total_dur - beats*shots*const_tr - prep_time_tot
        waittimes = divide_into_random_floats(waittime_tot, beats-1)
    else:
        waittimes = [max(0, total_dur/beats-ti[ii]-t2te[ii]-tsl[ii]-const_tr*shots) for ii in range(beats)]

    fa = np.repeat(random.choices((const_fa), k=beats), shots) if type(const_fa) == list else np.full(beats*shots, const_fa)

    # # Ensure that not all flipangles are zero
    # if np.count_nonzero(fa) == 0:
    #     return

    tr = np.full(beats*shots, 0)

    for ii in range(len(waittimes)):
        tr[(ii + 1) * shots - 1] += waittimes[ii] * 1e3

    ph = phase_inc * np.arange(n_ex).cumsum()

    mrf_sequence = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, const_tr, te, tsl)

    mrf_sequence.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1, t1rho=target_t1rho)

    return mrf_sequence


def optimize_sequence_worker(junk_size, args):
    
    sequences = []

    for _ in range(junk_size):
        
        # beats = random.randint(min_beats, max_beats)
        # n_ex = beats * shots

        # prep_order = random.choices(prep_modules, weights=prep_module_weights, k=beats)
        # prep = [BLOCKS[name]['prep'] for name in prep_order]
        # ti = [BLOCKS[name]['ti'] for name in prep_order]
        # t2te = [BLOCKS[name]['t2te'] for name in prep_order]
        # tsl = [BLOCKS[name]['tsl'] for name in prep_order]

        # prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] + BLOCKS[name]['tsl'] for name in prep_order])

        # if optimize_positions:
        #     waittime_tot = total_dur - beats*shots*const_tr - prep_time_tot
        #     waittimes = divide_into_random_floats(waittime_tot, beats-1)
        # else:
        #     waittimes = [total_dur/beats-ti[ii]-t2te[ii]-tsl[ii] for ii in range(beats)]

        # fa = np.repeat(random.choices((const_fa), k=beats), shots) if type(const_fa) == list else np.full(beats*shots, const_fa)

        # # Ensure that not all flipangles are zero
        # if np.count_nonzero(fa) == 0:
        #     continue

        # tr = np.full(beats*shots, 0)

        # for ii in range(len(waittimes)):
        #     tr[(ii + 1) * shots - 1] += waittimes[ii] * 1e3

        # ph = phase_inc * np.arange(n_ex).cumsum()

        # mrf_sequence = MRFSequence(beats, shots, fa, tr, ph, prep, ti, t2te, const_tr, te)

        # mrf_sequence.calc_cost(costfunction, target_t1, target_t2, target_m0, inv_eff, delta_B1)

        sequence = create_sequence(args)

        sequences.append(sequence)

    return sequences

def optimize_sequence(costfunction, target_t1, target_t2, target_t1rho, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights=None, min_beats=1, max_beats=None, n_iter_max=1e6, inv_eff=0.95, delta_B1=1., phase_inc=0., optimize_positions=True, parallel=True, num_workers=8, num_junks=1e2):

    sequences = []

    max_prep_dur = max([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] + BLOCKS[name]['tsl'] for name in prep_modules])

    if max_beats is None:
        max_beats = total_dur // (max_prep_dur + shots * const_tr)

    if min_beats is None:
        min_beats = max_beats

    print(f'Total sequence duration: {total_dur:.0f} ms.\nMax num of preps: {max_beats:.0f}.')

    args = costfunction, target_t1, target_t2, target_t1rho, target_m0, shots, const_fa, const_tr, te, total_dur, prep_modules, prep_module_weights, min_beats, max_beats, inv_eff, delta_B1, phase_inc, optimize_positions

    t0 = datetime.now()

    if parallel:
        junk_size = int(n_iter_max/num_junks)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            
            futures = [executor.submit(optimize_sequence_worker, junk_size, args) for _ in tqdm(range(int(num_junks)), desc='Setting up jobs')]
            
            for future in tqdm(as_completed(futures), total=int(num_junks), desc='Computing'):
                sequences.extend(future.result())
        
        timediff = datetime.now() - t0
        print(f'Total time: {str(timediff).split(".")[0]}. {len(sequences) / timediff.total_seconds():.2f} its/s.')

    else:
        count = 0

        try:
            while True:
                sequences.append(create_sequence(args))
                count += 1
                timediff = datetime.now()-t0
                print(f'{count} iters. Time: {str(timediff).split(".")[0]}. {count/timediff.total_seconds():.2f} its/s.', end='\r')

                if count >= n_iter_max:
                    break

        except KeyboardInterrupt:
            pass

    return sequences
