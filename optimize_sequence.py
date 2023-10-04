import numpy as np
import random
from datetime import datetime

from abdominal_tools import BLOCKS, divide_into_random_integers, MRFSequence


# def optimize_sequence(target_tissue, acq_block, mrf_sequence_ref, prep_modules, weightingmatrix, num_acq_blocks=12, avg_dur_block=1200, track_crlbs=True, store_good=20, store_bad=20):
def optimize_sequence(target_tissue, acq_block, mrf_sequence_ref, prep_modules, weightingmatrix, num_acq_blocks=12, avg_dur_block=1200):

    # crlbs, crlbs_T1, crlbs_T2, crlbs_M0 = [], [], [], []

    # best_sequences, worst_sequences = [], []

    mrf_sequence_ref.calc_crlb(target_tissue)
    ref_value = np.trace(weightingmatrix @ mrf_sequence_ref.crlb)

    sequences = []

    count = 0

    t0 = datetime.now()
    timestamp = t0.strftime('%Y%m%d_%H%M%S')[2:]

    try:
        while True:

            prep_order = random.choices(prep_modules, k=num_acq_blocks)

            prep_time_tot = sum([BLOCKS[name]['ti'] + BLOCKS[name]['t2te'] for name in prep_order])

            waittime_tot = int(num_acq_blocks*avg_dur_block - num_acq_blocks*sum(acq_block.tr) - prep_time_tot)

            waittimes = divide_into_random_integers(waittime_tot, num_acq_blocks)

            mrf_sequence = MRFSequence(prep_order, waittimes, acq_block)

            mrf_sequence.calc_crlb(target_tissue)

            sequences.append(mrf_sequence)

            sequences.sort(key = lambda x: np.trace(weightingmatrix @ x.crlb))



            # CRLB = weightingmatrix @ self.crlb

            # if track_crlbs:
            #     crlbs.append(np.trace(CRLB))    
            #     crlbs_T1.append(CRLB[0, 0])    
            #     crlbs_T2.append(CRLB[1, 1])    
            #     crlbs_M0.append(CRLB[2, 2])    
            

            # if len(best_sequences) < store_good:
            #     best_sequences.append(mrf_sequence)
            #     best_sequences.sort(key=lambda x: np.trace(weightingmatrix @ x.crlb))

            # elif mrf_sequence.crlb < best_sequences[-1].crlb:
            #     best_sequences[-1] = mrf_sequence
            #     best_sequences.sort(key=lambda x: np.trace(weightingmatrix @ x.crlb))


            # if len(worst_sequences) < store_bad:
            #     worst_sequences.append(mrf_sequence)
            #     worst_sequences.sort(key=lambda x: np.trace(weightingmatrix @ x.crlb))

            # elif mrf_sequence.crlb > worst_sequences[0].crlb:
            #     worst_sequences[0] = mrf_sequence
            #     worst_sequences.sort(key=lambda x: np.trace(weightingmatrix @ x.crlb))

            count += 1

            # print(f'{count} iters. Min CRLB: {best_sequences[0].crlb:.3f}. Impr of {(1-best_sequences[0].crlb/mrf_sequence_ref.crlb)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')
            print(f'{count} iters. Min CRLB: {np.trace(weightingmatrix@sequences[0].crlb):.3f}. Impr of {(1-np.trace(weightingmatrix@sequences[0].crlb)/ref_value)*100:.2f}%. Time: {str(datetime.now()-t0)}\t', end='\r')


    except KeyboardInterrupt:

        duration = str(datetime.now()-t0)

        # print(f'\n\nMin CRLB: {best_sequences[0].crlb:.3f}.')
        # print(f'Max CRLB: {worst_sequences[-1].crlb:.3f}.')
        # print(f'Ref CRLB: {mrf_sequence_ref.crlb:.3f}.')
        # print(f'Improvement of {(1-best_sequences[0].crlb/mrf_sequence_ref.crlb)*100:.2f}%.')
        # print(f'Time elapsed: {duration}.')

        # crlb_array = np.transpose(sorted(zip(crlbs, crlbs_T1, crlbs_T2, crlbs_M0)))

    # return count, best_sequences, worst_sequences, crlb_array, timestamp, duration    
    return count, sequences, timestamp, duration