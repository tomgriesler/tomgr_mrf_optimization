import numpy as np
import random
from datetime import datetime

from abdominal_tools import BLOCKS, divide_into_random_integers, MRFSequence


def optimize_sequence(target_tissue, acq_block, prep_modules, num_acq_blocks=12, avg_dur_block=1200, weightingmatrix=None, ref_crlb_matrix=None, update_every=10, N_iter_max=np.inf):

    sequences = []

    ref_value = np.trace(weightingmatrix@ref_crlb_matrix) if ref_crlb_matrix is not None else False

    count = 0

    t0 = datetime.now()

    try:
        while True:

            prep_order = random.choices(prep_modules, k=num_acq_blocks)

            prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_order])

            waittime_tot = int(num_acq_blocks*avg_dur_block - num_acq_blocks*sum(acq_block.tr) - prep_time_tot)

            waittimes = divide_into_random_integers(waittime_tot, num_acq_blocks)

            mrf_sequence = MRFSequence(prep_order, waittimes, acq_block)

            mrf_sequence.calc_crlb(target_tissue)

            sequences.append(mrf_sequence)

            count += 1

            if count%update_every==0:
                    
                if ref_value:    
                    sequences.sort(key = lambda x: np.trace(weightingmatrix @ x.crlb))
                    print(f'{count} iters. Min CRLB: {np.trace(weightingmatrix@sequences[0].crlb):.3f}. Impr of {(1-np.trace(weightingmatrix@sequences[0].crlb)/ref_value)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')
                
                else: 
                    print(f'{count} iters. Time: {str(datetime.now()-t0)}\t', end='\r')

            if count >= N_iter_max:
                break


    except KeyboardInterrupt:
        pass
        
    return sequences