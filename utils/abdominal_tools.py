import numpy as np
import random
from tqdm import tqdm

from signalmodel_fisp_epg_numpy import calculate_signal_fisp, calculate_crlb_fisp, calculate_crlb_fisp_pv, calculate_orthogonality

from signalmodel_bssfp_numpy import calculate_signal_bssfp


def divide_into_random_floats(N, n):

    positions = [0] + sorted([random.uniform(0, N) for _ in range(n-1)]) + [N]
    floats = [positions[ii+1]-positions[ii] for ii in range(n)]

    return floats


def create_weightingmatrix(weighting, target_t1=np.inf, target_t2=np.inf, target_t1rho=np.inf, dims=3):

    if target_t1rho == None:
        target_t1rho = np.inf

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
        
    def calc_signal_fisp(self, t1, t2, m0, inv_eff=0.95, delta_B1=1., t1rho=None, return_result=False):

        signal = calculate_signal_fisp(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho)

        if return_result:
            return signal
        else:
            self.signal = signal

    def calc_signal_bssfp(self, t1, t2, m0, inv_eff=0.95, delta_B1=1., df=0, t1rho=None, return_result=False):

        signal = calculate_signal_bssfp(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, df, t1rho)

        if return_result:
            return signal
        else:
            self.signal = signal

    def calc_cost(self, costfunction, t1, t2, m0, inv_eff=0.95, delta_B1=1., fraction=None, t1rho=None, return_result=False):

        if costfunction == 'crlb':

            if type(t1)==list or type(t2)==list:
                raise TypeError('Only enter relaxation times of one tissue.')

            v = calculate_crlb_fisp(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho)
            cost = np.sqrt(np.diagonal(v))

        elif costfunction == 'orthogonality': 

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')

            cost = calculate_orthogonality(t1, t2, m0, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1, t1rho)

        elif costfunction == 'crlb_pv':

            if type(t1)!=list or type(t2)!=list:
                raise TypeError('Enter relaxation times of two tissues.')
            
            cost = np.sqrt(calculate_crlb_fisp_pv(t1, t2, m0, fraction, self.beats, self.shots, self.fa, self.tr, self.ph, self.prep, self.ti, self.t2te, self.tr_offset, self.te, inv_eff, delta_B1))

        else: 
            raise ValueError('Not a valid costfunction.')
        
        if return_result:
            return cost
        else:
            self.cost = cost