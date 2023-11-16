import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

from signalmodel_abdominal import calculate_signal_abdominal, calculate_crlb_abdominal


RESULTSPATH = Path('/home/tomgr/Documents/abdominal/results_optim')


BLOCKS = {
    'noPrep': {'prep': 0, 'ti': 0, 't2te': 0},
    'ti12': {'prep': 1, 'ti': 12, 't2te': 0},
    'ti21': {'prep': 1, 'ti': 21, 't2te': 0},
    'ti100': {'prep': 1, 'ti': 100, 't2te': 0},
    'ti250': {'prep': 1, 'ti': 250, 't2te': 0},
    'ti300': {'prep': 1, 'ti': 300, 't2te': 0},
    'ti400': {'prep': 1, 'ti': 400, 't2te': 0},
    't2prep40': {'prep': 2, 'ti': 0, 't2te': 40},
    't2prep50': {'prep': 2, 'ti': 0, 't2te': 50},
    't2prep80': {'prep': 2, 'ti': 0, 't2te': 80},
    't2prep120': {'prep': 2, 'ti': 0, 't2te': 120},
    't2prep160': {'prep': 2, 'ti': 0, 't2te': 160}
}


def divide_into_random_integers(N, n):

    positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
    integers = [positions[i+1]-positions[i] for i in range(n)]

    return integers


'''
TODO refactor
'''
def visualize_sequence(mrf_sequence, acq_block):

    prep_pulse_timings = [i*sum(acq_block.tr) + sum(mrf_sequence.ti[:i]) + sum(mrf_sequence.t2te[:i]) + sum(mrf_sequence.waittimes[:i]) for i in range(len(mrf_sequence.prep))]

    map = {
        0: {'color': 'white', 'label': None},
        1: {'color': 'tab:blue', 'label': 't1 prep'},
        2: {'color': 'tab:red', 'label': 't2 prep'}
    }

    for i, prep in enumerate(mrf_sequence.prep):
        prep_length = mrf_sequence.ti[i] + mrf_sequence.t2te[i]
        plt.axvspan(prep_pulse_timings[i], prep_pulse_timings[i]+prep_length, color=map[prep]['color'], label=map[prep]['label'], alpha=1)
        plt.axvline(prep_pulse_timings[i], color=map[prep]['color'])
        plt.axvline(prep_pulse_timings[i]+prep_length, color=map[prep]['color'])
        plt.axvspan(prep_pulse_timings[i]+prep_length, prep_pulse_timings[i]+prep_length+sum(acq_block.tr), color='gray', alpha=0.2, label='acquisition')


def visualize_crlb(sequences, weightingmatrix):

    crlbs = np.array([np.multiply(weightingmatrix, sequence.crlb) for sequence in sequences])

    if weightingmatrix[0]:
        plt.plot(crlbs[:, 0], '.', label='$cost_1$', alpha=0.5, ms=0.1, color='tab:blue')
    if weightingmatrix[1]:
        plt.plot(crlbs[:, 1], '.', label='$cost_2$', alpha=0.5, ms=0.1, color='tab:red')
    if weightingmatrix[0] and weightingmatrix[1]:
        plt.plot(np.sum(crlbs, axis=1), '.', label='$cost_3$', ms=0.1, color='tab:green')


def create_weightingmatrix(target_tissue, weighting):

    WEIGHTINGMATRICES = {
        '1, 1, 0': np.array([1, 1, 0]),
        '1/t1, 0, 0': np.array([1/target_tissue.t1, 0, 0]),
        '0, 1/t2, 0': np.array([0, 1/target_tissue.t2, 0]),
        '1/t1, 1/t2, 0': np.array([1/target_tissue.t1, 1/target_tissue.t2, 0]),
        '1/t1, 1/t2, 1/m0': np.array([1/target_tissue.t1, 1/target_tissue.t2, 1/target_tissue.m0]),
        '1/t1**2, 1/t2**2, 0': np.array([1/target_tissue.t1**2, 1/target_tissue.t2**2, 0]),
        '1/t1**2, 1/t2**2, 1/m0**2': np.array([1/target_tissue.t1**2, 1/target_tissue.t2**2, 1/target_tissue.m0**2])
    }
        
    return WEIGHTINGMATRICES[weighting]


def sort_sequences(sequences, weightingmatrix):

    sequences.sort(key = lambda x: np.sum(np.multiply(weightingmatrix, x.crlb)))


def store_optimization(sequences, prot, fa, tr):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    resultspath = RESULTSPATH/timestamp
    resultspath.mkdir()

    with open(resultspath/'sequences.pkl', 'wb') as handle:
        pickle.dump(sequences, handle)

    with open(resultspath/'prot.json', 'w') as handle: 
        json.dump(prot, handle, indent='\t')

    np.savetxt(resultspath/'FA.txt', fa)
    np.savetxt(resultspath/'TR.txt', tr)


class TargetTissue:

    def __init__(self, t1, t2, m0):
        self.t1 = t1
        self.t2 = t2
        self.m0 = m0


class MRFSequence:

    def __init__(self, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te):
        self.beats = beats
        self.shots = shots
        self.fa = np.array(fa)
        self.tr = np.array(tr)
        self.ph = np.array(ph)
        self.prep = np.array(prep)
        self.ti = np.array(ti)
        self.t2te = np.array(t2te)
        self.tr_offset = tr_offset
        self.te = te
        
    def calc_signal(self, t1, t2, m0, inversion_efficiency=0.95, delta_B1=1.):

        self.signal = calculate_signal_abdominal(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inversion_efficiency, delta_B1)

    def calc_crlb(self, t1, t2, m0, inversion_efficiency=0.95, delta_B1=1., sigma=1.):

        v = calculate_crlb_abdominal(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inversion_efficiency, delta_B1, sigma)

        self.crlb = np.sqrt(np.diagonal(v))