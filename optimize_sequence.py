import numpy as np
import random
import json
from datetime import datetime
import pickle
import shutil

from abdominal_tools import RESULTSPATH, BLOCKS, divide_into_random_integers, MRFSequence, AcquisitionBlock


def return_reference(reference):

    acq_block_fa = np.load('/home/tomgr/Documents/code/abdominal/fa_jaubert.npy')
    acq_block_tr = np.load('/home/tomgr/Documents/code/abdominal/tr_jaubert.npy')

    if reference == 'jaubert':

        prep_order_ref = ['TI12', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI300', 'noPrep', 'T2prep40', 'T2prep80', 'T2prep160', 'TI12', 'noPrep']

    elif reference == 'hamilton': 

        prep_order_ref = ['TI21', 'noPrep', 'T2prep40', 'T2prep80', 'TI100', 'noPrep', 'T2prep40', 'T2prep80', 'TI250', 'noPrep', 'T2prep40', 'T2prep80', 'TI400', 'noPrep', 'T2prep40', 'T2prep80']


    waittimes_ref = [1.2e3 - sum(BLOCKS[name]['tr']) - sum(acq_block_tr[:-1]) for name in prep_order_ref]

    acquisition_block = AcquisitionBlock(acq_block_fa, acq_block_tr, 1.4)

    mrf_sequence_ref = MRFSequence(prep_order_ref, waittimes_ref, acq_block_fa, acq_block_tr, BLOCKS)

    return acquisition_block, mrf_sequence_ref


def optimize_sequence(target_tissue, acquisition_block, mrf_sequence_ref, prep_modules, num_acq_blocks=12, track_crlbs=True, store_good=20, store_bad=20):

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

            mrf_sequence.calc_crlb(target_tissue, acquisition_block.TE)

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


def store_optimization(count, best_sequences, worst_sequences, crlb_array, timestamp, duration, mrf_sequence_ref, target_tissue, acquisition_block, reference):

    resultspath = RESULTSPATH / timestamp
    resultspath.mkdir()

    with open(resultspath/'mrf_sequence_ref.pkl', 'wb') as handle:
        pickle.dump(mrf_sequence_ref, handle)

    prot = {
        'count': count,
        'crlb_min': best_sequences[0].crlb,
        'reduction': 1-best_sequences[0].crlb/mrf_sequence_ref.crlb,
        'duration': duration,
        'reference': reference
    }
    with open(resultspath/'prot.json', 'w') as handle:
        json.dump(prot, handle, indent='\t')

    np.save(resultspath/'crlb_array.npy', crlb_array)

    # shutil.copy('/home/tomgr/Documents/code/abdominal/optimize_sequence.py', resultspath/'optimize_sequence_copy.py')

    with open(resultspath/'blocks.json', 'w') as handle:
        json.dump(BLOCKS, handle, indent='\t')

    with open(resultspath/'best_sequences.pkl', 'wb') as handle:
        pickle.dump({i: best_sequences[i] for i in range(len(best_sequences))}, handle)
    with open(resultspath/'worst_sequences.pkl', 'wb') as handle:
        pickle.dump({i: best_sequences[i] for i in range(len(worst_sequences))}, handle)

    with open(resultspath/'acquisition_block.pkl', 'wb') as handle:
        pickle.dump(acquisition_block, handle)
    with open(resultspath/'target_tissue.pkl', 'wb') as handle:
        pickle.dump(target_tissue, handle)

