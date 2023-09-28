import numpy as np
import random
from datetime import datetime

from abdominal_tools import BLOCKS, divide_into_random_integers, MRFSequence


def optimize_sequence(target_tissue, acquisition_block, mrf_sequence_ref, prep_modules, weightingmatrix=None, num_acq_blocks=12, track_crlbs=True, store_good=20, store_bad=20):

    crlbs, crlbs_T1, crlbs_T2, crlbs_M0 = [], [], [], []

    best_sequences, worst_sequences = [], []

    count = 0

    t0 = datetime.now()
    timestamp = t0.strftime('%Y%m%d_%H%M')[2:]

    try:
        while True:

            prep_order = random.choices(prep_modules, k=num_acq_blocks)
            prep_time_tot = sum(np.concatenate([BLOCKS[module]['tr'] for module in prep_order]))
            waittime_tot = int(sum(mrf_sequence_ref.tr) - num_acq_blocks*sum(acquisition_block.tr[:-1]) - prep_time_tot)
            waittimes = divide_into_random_integers(waittime_tot, num_acq_blocks)

            mrf_sequence = MRFSequence(prep_order, waittimes, acquisition_block.fa, acquisition_block.tr, BLOCKS)

            mrf_sequence.calc_crlb(target_tissue, acquisition_block.TE, weightingmatrix=weightingmatrix)

            if track_crlbs:
                crlbs.append(mrf_sequence.crlb)    
                crlbs_T1.append(mrf_sequence.crlb_T1)    
                crlbs_T2.append(mrf_sequence.crlb_T2)    
                crlbs_M0.append(mrf_sequence.crlb_M0)    
            

            if len(best_sequences) < store_good:
                best_sequences.append(mrf_sequence)
                best_sequences.sort(key=lambda x: x.crlb)

            elif mrf_sequence.crlb < best_sequences[-1].crlb:
                best_sequences[-1] = mrf_sequence
                best_sequences.sort(key=lambda x: x.crlb)


            if len(worst_sequences) < store_bad:
                worst_sequences.append(mrf_sequence)
                worst_sequences.sort(key=lambda x: x.crlb)

            elif mrf_sequence.crlb > worst_sequences[0].crlb:
                worst_sequences[0] = mrf_sequence
                worst_sequences.sort(key=lambda x: x.crlb)

            count += 1

            print(f'{count} iters. Min CRLB: {best_sequences[0].crlb:.3f}. Impr of {(1-best_sequences[0].crlb/mrf_sequence_ref.crlb)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')


    except KeyboardInterrupt:

        duration = str(datetime.now()-t0)

        print(f'\n\nMin CRLB: {best_sequences[0].crlb:.3f}.')
        print(f'Max CRLB: {worst_sequences[-1].crlb:.3f}.')
        print(f'Ref CRLB: {mrf_sequence_ref.crlb:.3f}.')
        print(f'Improvement of {(1-best_sequences[0].crlb/mrf_sequence_ref.crlb)*100:.2f}%.')
        print(f'Time elapsed: {duration}.')

        crlb_array = np.transpose(sorted(zip(crlbs, crlbs_T1, crlbs_T2, crlbs_M0)))

    return count, best_sequences, worst_sequences, crlb_array, timestamp, duration