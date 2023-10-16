import numpy as np
import random
from datetime import datetime

from abdominal_tools import BLOCKS, divide_into_random_integers, MRFSequence


def optimize_sequence(target_tissue, acq_block, prep_modules, total_dur, min_num_preps=1, N_iter_max=np.inf, inversion_efficiency=0.95, delta_B1=1):

    sequences = []

    count = 0

    max_prep_dur = max([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_modules])

    max_num_preps = total_dur // (max_prep_dur+sum(acq_block.tr))
    print(f'Total sequence duration: {total_dur:.0f} ms.\nMax num of preps: {max_num_preps:.0f}.')

    t0 = datetime.now()

    try:
        while True:

            num_acq_blocks = random.randint(min_num_preps, max_num_preps)

            prep_order = random.choices(prep_modules, k=num_acq_blocks)

            prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_order])

            waittime_tot = int(total_dur - num_acq_blocks*sum(acq_block.tr) - prep_time_tot)

            waittimes = divide_into_random_integers(waittime_tot, num_acq_blocks)

            mrf_sequence = MRFSequence(prep_order, waittimes)

            mrf_sequence.calc_crlb(acq_block, target_tissue, inversion_efficiency, delta_B1)

            sequences.append(mrf_sequence)

            count += 1

            timediff = datetime.now()-t0

            print(f'{count} iters. Time: {str(timediff).split(".")[0]}. {count/timediff.total_seconds():.2f} its/s.', end='\r')

            if count >= N_iter_max:
                break

    except KeyboardInterrupt:
        pass
        
    return sequences