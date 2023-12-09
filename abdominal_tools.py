import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
import subprocess
from tqdm import tqdm


from signalmodel_abdominal import calculate_signal, calculate_crlb, calculate_crlb_pv, calculate_orthogonality


RESULTSPATH = Path('/home/tomgr/Documents/abdominal/data/sequences')


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
    integers = [positions[ii+1]-positions[ii] for ii in range(n)]

    return integers


def visualize_sequence(mrf_sequence, show_fa=False):
    
    prep_pulse_timings = [ii*mrf_sequence.shots*mrf_sequence.tr_offset+np.sum(mrf_sequence.tr[:ii*mrf_sequence.shots])*1e-3+np.sum(mrf_sequence.ti[:ii])+np.sum(mrf_sequence.t2te[:ii]) for ii in range(mrf_sequence.beats)]

    map = {
        0: {'color': 'white', 'label': None},
        1: {'color': 'tab:blue', 'label': 'T1 prep'},
        2: {'color': 'tab:red', 'label': 'T2 prep'}
    }

    for ii in range(mrf_sequence.beats): 
        prep_length = mrf_sequence.ti[ii] + mrf_sequence.t2te[ii]
        plt.axvspan(prep_pulse_timings[ii], prep_pulse_timings[ii]+prep_length, color=map[mrf_sequence.prep[ii]]['color'], label=map[mrf_sequence.prep[ii]]['label'], alpha=1)
        plt.axvline(prep_pulse_timings[ii], color=map[mrf_sequence.prep[ii]]['color'])
        plt.axvline(prep_pulse_timings[ii]+prep_length, color=map[mrf_sequence.prep[ii]]['color'])
        plt.axvspan(prep_pulse_timings[ii]+prep_length, prep_pulse_timings[ii]+prep_length+sum(mrf_sequence.tr[ii*mrf_sequence.shots:(ii+1)*mrf_sequence.shots-1])*1e-3+mrf_sequence.shots*mrf_sequence.tr_offset, color='gray', alpha=0.2, label='acquisition')

        if show_fa:
            plt.plot([prep_pulse_timings[ii]+prep_length+jj*mrf_sequence.tr_offset for jj in range(mrf_sequence.shots)], mrf_sequence.fa[ii*mrf_sequence.shots:(ii+1)*mrf_sequence.shots], 'o', color='black', ms=2)


def visualize_cost(sequences, weightingmatrix):

    crlbs = np.array([np.multiply(weightingmatrix, sequence.cost) for sequence in sequences])

    if weightingmatrix[0]:
        plt.plot(crlbs[:, 0], '.', label='$cost_1$', alpha=0.5, ms=0.1, color='tab:blue')
    if weightingmatrix[1]:
        plt.plot(crlbs[:, 1], '.', label='$cost_2$', alpha=0.5, ms=0.1, color='tab:red')
    if weightingmatrix[0] and weightingmatrix[1]:
        plt.plot(np.sum(crlbs, axis=1), '.', label='$cost_3$', ms=0.1, color='tab:green')


def create_weightingmatrix(target_t1, target_t2, target_m0, weighting):

    WEIGHTINGMATRICES = {
        '1, 1, 0': np.array([1, 1, 0]),
        '1/T1, 0, 0': np.array([1/target_t1, 0, 0]),
        '0, 1/T2, 0': np.array([0, 1/target_t2, 0]),
        '1/T1, 1/T2, 0': np.array([1/target_t1, 1/target_t2, 0]),
        '1/T1, 1/T2, 1/M0': np.array([1/target_t1, 1/target_t2, 1/target_m0]),
        '1/T1**2, 1/T2**2, 0': np.array([1/target_t1**2, 1/target_t2**2, 0]),
        '1/T1**2, 1/T2**2, 1/M0**2': np.array([1/target_t1**2, 1/target_t2**2, 1/target_m0**2])
    }
        
    return WEIGHTINGMATRICES[weighting]


def sort_sequences(sequences, weightingmatrix=None):

    if weightingmatrix is not None:
        sequences.sort(key = lambda x: np.sum(np.multiply(weightingmatrix, x.cost)))

    else: 
        sequences.sort(key=lambda x: x.cost)


def store_optimization(resultspath, sequences, prot):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    timestamppath = resultspath/timestamp
    timestamppath.mkdir()

    for sequence in tqdm(sequences, desc='Compressing', total=len(sequences)):
        sequence.compress()

    print('Saving...', end='')
    
    with open(timestamppath/'sequences.pkl', 'wb') as handle:
        pickle.dump(sequences, handle)

    with open(timestamppath/'prot.json', 'w') as handle: 
        json.dump(prot, handle, indent='\t')

    print('done.')


def get_githash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_gitbranch() -> str:
    return str(subprocess.check_output(['git', 'branch'])).split("* ")[1].split("\\n")[0]


class TargetTissue:

    def __init__(self, t1, t2, m0):
        self.t1 = t1
        self.t2 = t2
        self.m0 = m0


class MRFSequence:

    def __init__(self, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te):
        self.beats = beats
        self.shots = shots
        self.fa = np.array(fa, dtype=np.float32)
        self.tr = np.array(tr, dtype=np.float32)
        self.ph = np.array(ph, dtype=np.float32)
        self.prep = np.array(prep, dtype=np.float32)
        self.ti = np.array(ti, dtype=np.float32)
        self.t2te = np.array(t2te, dtype=np.float32)
        self.tr_offset = tr_offset
        self.te = te

    def compress(self):
        self.fa_compressed = [self.fa[ii] for ii in np.arange(self.beats)*self.shots]
        self.tr_indices = np.nonzero(self.tr)
        self.tr_compressed = self.tr[self.tr_indices]
        self.ph_inc = self.ph[1]
        delattr(self, 'fa')
        delattr(self, 'tr')    
        delattr(self, 'ph')

    def decompress(self):
        self.fa = np.repeat(self.fa_compressed, self.shots)
        self.tr = np.zeros_like(self.fa)
        self.tr[self.tr_indices] = self.tr_compressed
        if not hasattr(self, 'ph'):
            self.ph = np.zeros_like(self.fa)
        self.ph = self.ph_inc*np.arange(len(self.fa)).cumsum()
        delattr(self, 'fa_compressed')
        delattr(self, 'tr_indices') 
        delattr(self, 'tr_compressed')  
        delattr(self, 'ph_inc')
        
    def calc_signal(self, t1, t2, m0, inv_eff=0.95, delta_B1=1.):

        self.signal = calculate_signal(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1)

    def calc_cost(self, costfunction, t1, t2, m0, inv_eff=0.95, delta_B1=1., fraction=None):

        if costfunction == 'crlb':

            if type(t1)==list or type(t2)==list:
                raise TypeError('Only enter relaxation times of one tissue.')

            v = calculate_crlb(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1)
            self.cost = np.sqrt(np.diagonal(v))

        elif costfunction == 'orthogonality': 

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')

            self.cost = calculate_orthogonality(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1)

        elif costfunction == 'crlb_pv':

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')
            
            self.cost = calculate_crlb_pv(t1, t2, m0, fraction, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1)

        else: 
            raise ValueError('Not a valid costfunction.')