import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
import subprocess
from tqdm import tqdm


from signalmodel_abdominal import calculate_signal, calculate_crlb, calculate_crlb_pv, calculate_orthogonality, calculate_crlb_orthogonality_combined


RESULTSPATH = Path('/scratch/abdominal/data/sequences')


BLOCKS = {
    'noPrep': {'prep': 0, 'ti': 0, 't2te': 0, 'tsl': 0},
    'TI12': {'prep': 1, 'ti': 12, 't2te': 0, 'tsl': 0},
    'TI21': {'prep': 1, 'ti': 21, 't2te': 0, 'tsl': 0},
    'TI100': {'prep': 1, 'ti': 100, 't2te': 0, 'tsl': 0},
    'TI250': {'prep': 1, 'ti': 250, 't2te': 0, 'tsl': 0},
    'TI300': {'prep': 1, 'ti': 300, 't2te': 0, 'tsl': 0},
    'TI400': {'prep': 1, 'ti': 400, 't2te': 0, 'tsl': 0},
    'T2prep30': {'prep': 2, 'ti': 0, 't2te': 30, 'tsl': 0},
    'T2prep40': {'prep': 2, 'ti': 0, 't2te': 40, 'tsl': 0},
    'T2prep50': {'prep': 2, 'ti': 0, 't2te': 50, 'tsl': 0},
    'T2prep80': {'prep': 2, 'ti': 0, 't2te': 80, 'tsl': 0},
    'T2prep120': {'prep': 2, 'ti': 0, 't2te': 120, 'tsl': 0},
    'T2prep160': {'prep': 2, 'ti': 0, 't2te': 160, 'tsl': 0},
    'T1rhoprep30': {'prep': 3, 'ti': 0, 't2te': 0, 'tsl': 30},
    'T1rhoprep50': {'prep': 3, 'ti': 0, 't2te': 0, 'tsl': 50},
    'T1rhoprep60': {'prep': 3, 'ti': 0, 't2te': 0, 'tsl': 60},
}


# def divide_into_random_integers(N, n):

#     positions = [0] + sorted(list(random.sample(range(1, N), n-1))) + [N]
#     integers = [positions[ii+1]-positions[ii] for ii in range(n)]

#     return integers


def divide_into_random_floats(N, n):

    positions = [0] + sorted([random.uniform(0, N) for _ in range(n-1)]) + [N]
    floats = [positions[ii+1]-positions[ii] for ii in range(n)]

    return floats


def visualize_sequence(mrf_sequence, show_fa=False):

    try:
        mrf_sequence.tsl
    except AttributeError:
        mrf_sequence.tsl = np.zeros_like(mrf_sequence.prep, dtype=np.float32)
    
    prep_pulse_timings = [ii*mrf_sequence.shots*mrf_sequence.tr_offset+np.sum(mrf_sequence.tr[:ii*mrf_sequence.shots])*1e-3+np.sum(mrf_sequence.ti[:ii])+np.sum(mrf_sequence.t2te[:ii])+np.sum(mrf_sequence.tsl[:ii]) for ii in range(mrf_sequence.beats)]

    map = {
        0: {'color': 'white', 'label': None},
        1: {'color': 'tab:red', 'label': 'T1 prep'},
        2: {'color': 'tab:blue', 'label': 'T2 prep'},
        3: {'color': 'tab:purple', 'label': 'T1rho prep'}
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
        plt.plot(crlbs[:, 0], '.', label='$cost_T1$', alpha=0.5, ms=0.1, color='tab:blue')
    if weightingmatrix[1]:
        plt.plot(crlbs[:, 1], '.', label='$cost_T2$', alpha=0.5, ms=0.1, color='tab:red')
    if weightingmatrix[3]:
        plt.plot(crlbs[:, 3], '.', label='$cost_T1rho$', alpha=0.5, ms=0.1, color='tab:red')
    if weightingmatrix[0] and weightingmatrix[1]:
        plt.plot(np.sum(crlbs, axis=1), '.', label='$cost_T1T2T1rho$', ms=0.1, color='tab:green')


def create_weightingmatrix(weighting, target_t1=np.inf, target_t2=np.inf, target_t1rho=np.inf, dims=3):

    WEIGHTINGMATRICES = {
        'T1': np.array([1/target_t1, 0, 0, 0][:dims]),
        'T2': np.array([0, 1/target_t2, 0, 0][:dims]),
        'T1rho': np.array([0, 0, 0, 1/target_t1rho][:dims]),
        'T1, T2': np.array([1/target_t1, 1/target_t2, 0, 0][:dims]),
        'T1, T2, T1rho': np.array([1/target_t1, 1/target_t2, 0, 1/target_t1rho][:dims])
    }
        
    return WEIGHTINGMATRICES[weighting]


def sort_sequences(sequences, weightingmatrix=None):

    cost_list = [np.sum(np.multiply(weightingmatrix, x.cost)) for x in tqdm(sequences, total=len(sequences), desc='Create cost list')] if weightingmatrix is not None else [x.cost for x in sequences]
    order = np.argsort(cost_list)
    sequences = [sequences[idx] for idx in tqdm(order, total=len(order), desc='Sort')]
    return sequences


def sort_sequences_inplace(sequences, weightingmatrix):

    sequences.sort(key = lambda x: np.sum(np.multiply(weightingmatrix, x.cost)))


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


class MRFSequence:

    def __init__(self, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, tsl=None):
        self.beats = beats
        self.shots = shots
        self.fa = np.array(fa, dtype=np.float32)
        self.tr = np.array(tr, dtype=np.float32)
        self.ph = np.array(ph, dtype=np.float32)
        self.prep = np.array(prep, dtype=np.float32)
        self.ti = np.array(ti, dtype=np.float32)
        self.t2te = np.array(t2te, dtype=np.float32)
        self.tsl = np.zeros_like(self.prep, dtype=np.float32) if tsl is None else np.array(tsl, dtype=np.float32)
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
        
    def calc_signal(self, t1, t2, m0, inv_eff=0.95, delta_B1=1., t1rho=None, return_result=False):

        signal = calculate_signal(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho, self.tsl)

        if return_result:
            return signal
        else:
            self.signal = signal

    def calc_cost(self, costfunction, t1, t2, m0, inv_eff=0.95, delta_B1=1., fraction=None, t1rho=None, return_result=False):

        if costfunction == 'crlb':

            if type(t1)==list or type(t2)==list:
                raise TypeError('Only enter relaxation times of one tissue.')

            v = calculate_crlb(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho, self.tsl)
            cost = np.sqrt(np.diagonal(v))

        elif costfunction == 'orthogonality': 

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')

            cost = calculate_orthogonality(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho, self.tsl)

        elif costfunction == 'crlb_pv':

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')
            
            cost = np.sqrt(calculate_crlb_pv(t1, t2, m0, fraction, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1))

        elif costfunction == 'crlb_orth':

            cost = calculate_crlb_orthogonality_combined(t1, t2, t1rho, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tsl, self.tr_offset, self.te, inv_eff, delta_B1)

        else: 
            raise ValueError('Not a valid costfunction.')
        
        if return_result:
            return cost
        else:
            self.cost = cost