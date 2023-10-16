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
    'T2prep50': {'prep': 2, 'ti': 0, 't2te': 50},
    'T2prep80': {'prep': 2, 'ti': 0, 't2te': 80},
    'T2prep120': {'prep': 2, 'ti': 0, 't2te': 120},
    'T2prep160': {'prep': 2, 'ti': 0, 't2te': 160}
}


def divide_into_random_integers(N, n):

    positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
    integers = [positions[i+1]-positions[i] for i in range(n)]

    return integers


def visualize_sequence(mrf_sequence, acq_block):

    prep_pulse_timings = [i*sum(acq_block.tr) + sum(mrf_sequence.TI[:i]) + sum(mrf_sequence.T2TE[:i]) + sum(mrf_sequence.waittimes[:i]) for i in range(len(mrf_sequence.PREP))]

    map = {
        0: {'color': 'gray', 'label': 'noPrep'},
        1: {'color': 'tab:blue', 'label': 'T1'},
        2: {'color': 'tab:red', 'label': 'T2'}
    }

    for i, prep in enumerate(mrf_sequence.PREP):
        prep_length = mrf_sequence.TI[i] + mrf_sequence.T2TE[i]
        plt.axvspan(prep_pulse_timings[i], prep_pulse_timings[i]+prep_length, color=map[prep]['color'], label=map[prep]['label'], alpha=0.5)
        plt.axvline(prep_pulse_timings[i], color=map[prep]['color'])
        plt.axvline(prep_pulse_timings[i]+prep_length, color=map[prep]['color'])
        plt.axvspan(prep_pulse_timings[i]+prep_length, prep_pulse_timings[i]+prep_length+sum(acq_block.tr), color='gray', alpha=0.2, label='acquisition')
        plt.plot(prep_pulse_timings[i]+prep_length+[sum(acq_block.tr[:i]) for i in range(len(acq_block.tr))], acq_block.fa, '.', color='black', label='excitation')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', ncols=2)

    plt.ylim(0, max(acq_block.fa)*4/3)

    plt.xlabel('Time [ms]')
    plt.ylabel('FA [deg]')


def visualize_crlb(sequences, weightingmatrix):

    crlbs = np.array([np.multiply(weightingmatrix, sequence.crlb) for sequence in sequences])

    plt.plot(np.sum(crlbs, axis=1), label='total')
    plt.plot(crlbs[:, 0], label='T1')
    plt.plot(crlbs[:, 1], label='T2')
    plt.plot(crlbs[:, 2], label='M0')
    plt.legend()

    plt.xlabel('N')
    plt.ylabel('rel. CRLB')


def create_weightingmatrix(target_tissue, weighting):

    WEIGHTINGMATRICES = {
        '1, 1, 0': np.array([1, 1, 0]),
        '1/T1, 0, 0': np.array([1/target_tissue.T1, 0, 0]),
        '0, 1/T2, 0': np.array([0, 1/target_tissue.T2, 0]),
        '1/T1, 1/T2, 0': np.array([1/target_tissue.T1, 1/target_tissue.T2, 0]),
        '1/T1, 1/T2, 1/M0': np.array([1/target_tissue.T1, 1/target_tissue.T2, 1/target_tissue.M0]),
        '1/T1**2, 1/T2**2, 0': np.array([1/target_tissue.T1**2, 1/target_tissue.T2**2, 0]),
        '1/T1**2, 1/T2**2, 1/M0**2': np.array([1/target_tissue.T1**2, 1/target_tissue.T2**2, 1/target_tissue.M0**2])
    }
        
    return WEIGHTINGMATRICES[weighting]


def sort_sequences(sequences, weightingmatrix):

    sequences.sort(key = lambda x: np.sum(np.multiply(weightingmatrix, x.crlb)))


def store_optimization(sequences, acq_block, prot):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    resultspath = RESULTSPATH/timestamp
    resultspath.mkdir()

    with open(resultspath/'sequences.pkl', 'wb') as handle:
        pickle.dump(sequences, handle)

    with open(resultspath/'acq_block.pkl', 'wb') as handle:
        pickle.dump(acq_block, handle)

    with open(resultspath/'prot.json', 'w') as handle: 
        json.dump(prot, handle, indent='\t')


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

    def __init__(self, prep_order, waittimes):
        self.waittimes = waittimes
        self.PREP = [BLOCKS[name]['prep'] for name in prep_order]
        self.TI = [BLOCKS[name]['ti'] for name in prep_order]
        self.T2TE = [BLOCKS[name]['t2te'] for name in prep_order]
        
    def calc_signal(self, acq_block, target_tissue, inversion_efficiency=0.95, delta_B1=1):

        self.signal = calculate_signal_abdominal(target_tissue.T1, target_tissue.T2, target_tissue.M0, acq_block.fa, acq_block.tr, self.PREP, self.TI, self.T2TE, self.waittimes, acq_block.TE, inversion_efficiency, delta_B1)

    def calc_crlb(self, acq_block, target_tissue, inversion_efficiency=0.95, delta_B1=1, sigma=1):

        V = calculate_crlb_abdominal(target_tissue.T1, target_tissue.T2, target_tissue.M0, acq_block.fa, acq_block.tr, self.PREP, self.TI, self.T2TE, self.waittimes, acq_block.TE, inversion_efficiency, delta_B1, sigma)

        self.crlb = np.sqrt(np.diagonal(V))