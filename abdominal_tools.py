import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

from signalmodel_abdominal import calculate_signal_abdominal, calculate_crlb_abdominal


RESULTSPATH = Path('/home/tomgr/Documents/code/abdominal/results_optim')


BLOCKS = {
    'noPrep': {'prep': 0, 'ti': 0, 't2te': 0},
    'TI12': {'prep': 1, 'ti': 12, 't2te': 0},
    'TI21': {'prep': 1, 'ti': 21, 't2te': 0},
    'TI100': {'prep': 1, 'ti': 100, 't2te': 0},
    'TI250': {'prep': 1, 'ti': 250, 't2te': 0},
    'TI300': {'prep': 1, 'ti': 300, 't2te': 0},
    'TI400': {'prep': 1, 'ti': 400, 't2te': 0},
    'T2prep40': {'prep': 2, 'ti': 0, 't2te': 40},
    'T2prep80': {'prep': 2, 'ti': 0, 't2te': 80},
    'T2prep120': {'prep': 2, 'ti': 0, 't2te': 120},
    'T2prep160': {'prep': 2, 'ti': 0, 't2te': 160}
}


def divide_into_random_integers(N, n):

    positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
    integers = [positions[i+1]-positions[i] for i in range(n)]

    return integers


def visualize_sequence(mrf_sequence):

    prep_pulse_timings = [i*sum(mrf_sequence.acq_block.tr)+sum([mrf_sequence.blocks[name]['ti']+mrf_sequence.blocks[name]['t2te'] for name in mrf_sequence.prep_order[:i]])+sum(mrf_sequence.waittimes[:i]) for i in range(len(mrf_sequence.prep_order))]

    colormap = {prep: color for prep, color in zip(BLOCKS.keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])}

    for i in range(len(mrf_sequence.prep_order)):
        plt.axvspan(prep_pulse_timings[i], prep_pulse_timings[i]+BLOCKS[mrf_sequence.prep_order[i]]['ti']+BLOCKS[mrf_sequence.prep_order[i]]['t2te'], color=colormap[mrf_sequence.prep_order[i]], label=mrf_sequence.prep_order[i], alpha=0.5)
        plt.axvline(prep_pulse_timings[i], color=colormap[mrf_sequence.prep_order[i]])
        plt.axvline(prep_pulse_timings[i]+BLOCKS[mrf_sequence.prep_order[i]]['ti']+BLOCKS[mrf_sequence.prep_order[i]]['t2te'], color=colormap[mrf_sequence.prep_order[i]])
        plt.axvspan(prep_pulse_timings[i]+BLOCKS[mrf_sequence.prep_order[i]]['ti']+BLOCKS[mrf_sequence.prep_order[i]]['t2te'], prep_pulse_timings[i]+BLOCKS[mrf_sequence.prep_order[i]]['ti']+BLOCKS[mrf_sequence.prep_order[i]]['t2te']+sum(mrf_sequence.acq_block.tr), color='gray', alpha=0.2, label='acquisition')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', ncols=3)

    plt.ylim(0, 20)

    plt.xlabel('Time [ms]')
    plt.ylabel('FA [deg]')


def create_weightingmatrix(target_tissue, weighting):

    if weighting == '1, 1, 0':
        weightingmatrix = np.diag([1, 1, 0])
    elif weighting == '1/T1, 1/T2, 0':
        weightingmatrix = np.diag([1/target_tissue.T1, 1/target_tissue.T2, 0])
    elif weighting == '1/T1**2, 1/T2**2, 1/M0**2':
        weightingmatrix = np.diag([1/target_tissue.T1**2, 1/target_tissue.T2**2, 1/target_tissue.M0**2])

    return weightingmatrix


def sort_sequences(sequences, weightingmatrix):

    sequences.sort(key = lambda x: np.trace(weightingmatrix @ x.crlb))


def store_optimization(sequences, prot):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    resultspath = RESULTSPATH/timestamp

    with open(resultspath/'sequences.pkl', 'wb') as handle:
        pickle.dump(sequences, handle)

    with open(resultspath/'prot.json', 'r') as handle: 
        json.dump(prot, handle)



    

    # prot = {
    #     'count': count,
    #     'crlb_min': best_sequences[0].crlb,
    #     'reduction': 1-best_sequences[0].crlb/mrf_sequence_ref.crlb,
    #     'duration': duration,
    #     'reference': reference
    # }
    # with open(resultspath/'prot.json', 'w') as handle:
    #     json.dump(prot, handle, indent='\t')

    # np.save(resultspath/'crlb_array.npy', crlb_array)

    # with open(resultspath/'blocks.json', 'w') as handle:
    #     json.dump(BLOCKS, handle, indent='\t')

    # with open(resultspath/'best_sequences.pkl', 'wb') as handle:
    #     pickle.dump({i: best_sequences[i] for i in range(len(best_sequences))}, handle)
    # with open(resultspath/'worst_sequences.pkl', 'wb') as handle:
    #     pickle.dump({i: worst_sequences[i] for i in range(len(worst_sequences))}, handle)

    # with open(resultspath/'acq_block.pkl', 'wb') as handle:
    #     pickle.dump(acq_block, handle)
    # with open(resultspath/'target_tissue.pkl', 'wb') as handle:
    #     pickle.dump(target_tissue, handle)

    # if weightingmatrix:
    #     np.save(resultspath/'weightingmatrix.npy', weightingmatrix)


class AcquisitionBlock:

    def __init__(self, fa, tr, TE):
        self.fa = fa
        self.tr = tr
        self.TE = TE

class TargetTissue:

    def __init__(self, T1, T2, M0):
        self.T1 = T1
        self.T2 = T2
        self.M0 = M0


class MRFSequence:

    def __init__(self, prep_order, waittimes, acq_block):
        self.prep_order = prep_order
        self.waittimes = waittimes
        self.acq_block = acq_block
        self.blocks = {name: BLOCKS[name] for name in np.unique(self.prep_order)}
        self.PREP = [self.blocks[name]['prep'] for name in self.prep_order]
        self.TI = [self.blocks[name]['ti'] for name in self.prep_order]
        self.T2TE = [self.blocks[name]['t2te'] for name in self.prep_order]
        
    def calc_signal(self, target_tissue, inversion_efficiency=0.95, delta_B1=1):

        self.signal = calculate_signal_abdominal(target_tissue.T1, target_tissue.T2, target_tissue.M0, self.acq_block.fa, self.acq_block.tr, self.PREP, self.TI, self.T2TE, self.waittimes, self.acq_block.TE, inversion_efficiency, delta_B1)

    def calc_crlb(self, target_tissue, inversion_efficiency=0.95, delta_B1=1, sigma=1):

        V = calculate_crlb_abdominal(target_tissue.T1, target_tissue.T2, target_tissue.M0, self.acq_block.fa, self.acq_block.tr, self.PREP, self.TI, self.T2TE, self.waittimes, self.acq_block.TE, inversion_efficiency, delta_B1, sigma)

        self.crlb = V