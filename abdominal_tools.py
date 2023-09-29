import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import json
import pickle

import sys
sys.path.append('/home/tomgr/Documents/tg_mrf_optimization')
from costfunctions import calculate_crlb_sc_epg
from signalmodel_epg import calculate_complex_signal_epg


RESULTSPATH = Path('/home/tomgr/Documents/code/abdominal/results_optim')

BLOCKS = {
    'noPrep':
    {
        'fa': [],
        'tr': []
    },
    'TI12': {
        'fa': [180],
        'tr': [12]
    },
    'TI21': {
        'fa': [180],
        'tr': [21]
    },
    'TI100':
    {
        'fa': [180],
        'tr': [100]
    },
    'TI250':
    {
        'fa': [180],
        'tr': [250]
    },
    'TI300':
    {
        'fa': [180],
        'tr': [300]
    },
    'TI400':
    {
        'fa': [180],
        'tr': [400]
    },
    'T2prep40':
    {
        'fa': [90, -90], 
        'tr': [40, 20]
    },
    'T2prep80':
    {
        'fa': [90, -90],
        'tr': [80, 20]
    },
    'T2prep160':
    {
        'fa': [90, -90],
        'tr': [160, 20]
    }
}


def divide_into_random_integers(N, n):

    positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
    integers = [positions[i+1]-positions[i] for i in range(n)]

    return integers


def visualize_sequence(mrf_sequence):

    prep_pulse_timings = [0]
    for i in range(len(mrf_sequence.prep_order)-1):
        prep_pulse_timing = prep_pulse_timings[-1] + sum(BLOCKS[mrf_sequence.prep_order[i]]['tr']) + sum(mrf_sequence.acq_block_tr[:-1]) + mrf_sequence.waittimes[i]
        prep_pulse_timings.append(prep_pulse_timing)

    colormap = {prep: color for prep, color in zip(BLOCKS.keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])}

    plt.plot([sum(mrf_sequence.tr[:i]) for i in range(len(mrf_sequence.tr))], mrf_sequence.fa, '.', color='black')

    for i in range(len(mrf_sequence.prep_order)):
        plt.axvspan(prep_pulse_timings[i], prep_pulse_timings[i]+sum(BLOCKS[mrf_sequence.prep_order[i]]['tr']), color=colormap[mrf_sequence.prep_order[i]], label=mrf_sequence.prep_order[i], alpha=0.5)
        plt.axvline(prep_pulse_timings[i], color=colormap[mrf_sequence.prep_order[i]])
        plt.axvline(prep_pulse_timings[i]+sum(BLOCKS[mrf_sequence.prep_order[i]]['tr']), color=colormap[mrf_sequence.prep_order[i]])
        plt.axvspan(prep_pulse_timings[i]+sum(BLOCKS[mrf_sequence.prep_order[i]]['tr']), prep_pulse_timings[i]+sum(BLOCKS[mrf_sequence.prep_order[i]]['tr'])+sum(mrf_sequence.acq_block_tr[:-1]), color='gray', alpha=0.2, label='acquisition')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', ncols=3)

    plt.ylim(0, 20)

    plt.xlabel('Time [ms]')
    plt.ylabel('FA [deg]')


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


def store_optimization(count, best_sequences, worst_sequences, crlb_array, timestamp, duration, mrf_sequence_ref, target_tissue, acquisition_block, reference, weightingmatrix=None):

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

    with open(resultspath/'blocks.json', 'w') as handle:
        json.dump(BLOCKS, handle, indent='\t')

    with open(resultspath/'best_sequences.pkl', 'wb') as handle:
        pickle.dump({i: best_sequences[i] for i in range(len(best_sequences))}, handle)
    with open(resultspath/'worst_sequences.pkl', 'wb') as handle:
        pickle.dump({i: worst_sequences[i] for i in range(len(worst_sequences))}, handle)

    with open(resultspath/'acquisition_block.pkl', 'wb') as handle:
        pickle.dump(acquisition_block, handle)
    with open(resultspath/'target_tissue.pkl', 'wb') as handle:
        pickle.dump(target_tissue, handle)

    if weightingmatrix:
        np.save(resultspath/'weightingmatrix.npy', weightingmatrix)


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

    def __init__(self, prep_order, waittimes, acq_block_fa, acq_block_tr, blocks):
        self.prep_order = prep_order
        self.waittimes = waittimes
        self.acq_block_fa = acq_block_fa
        self.acq_block_tr = acq_block_tr
        self.fa = np.concatenate([np.concatenate([blocks[name]['fa'], self.acq_block_fa]) for name in self.prep_order])
        self.tr = np.concatenate([np.concatenate([blocks[name]['tr'], self.acq_block_tr[:-1], [wait_time]]) for name, wait_time in zip(self.prep_order, self.waittimes)])

    def calc_signal(self, target_tissue, TE):
        signal_temp = calculate_complex_signal_epg(target_tissue.T1, target_tissue.T2, target_tissue.M0, self.fa, self.tr, TE).detach().numpy()
        self.signal = signal_temp[np.isin(self.fa, self.acq_block_fa)]


        # self.signal = calculate_complex_signal_epg(target_tissue.T1, target_tissue.T2, target_tissue.M0, self.fa, self.tr, TE).detach().numpy()

    def calc_crlb(self, target_tissue, TE, weightingmatrix=None):
        V = calculate_crlb_sc_epg(target_tissue.T1, target_tissue.T2, target_tissue.M0, self.fa, self.tr, TE, return_crlb_matrix=True, weightingmatrix=weightingmatrix)
        self.crlb_T1 = V[0, 0].item()
        self.crlb_T2 = V[1, 1].item()
        self.crlb_M0 = V[2, 2].item()
        self.crlb = self.crlb_T1 + self.crlb_T2 + self.crlb_M0