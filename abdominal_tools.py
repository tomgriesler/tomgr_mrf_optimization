import numpy as np
from pathlib import Path
import random

import sys
sys.path.append('/home/tomgr/Documents/tg_mrf_optimization')
from costfunctions import calculate_crlb_sc_epg


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
    'TI300':
    {
        'fa': [180],
        'tr': [300]
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


class MRFSequence:

    def __init__(self, prep_order, waittimes, acq_block_fa, acq_block_tr, blocks):
        self.prep_order = prep_order
        self.waittimes = waittimes
        self.acq_block_fa = acq_block_fa
        self.acq_block_tr = acq_block_tr
        self.fa = np.concatenate([np.concatenate([blocks[name]['fa'], self.acq_block_fa]) for name in self.prep_order])
        self.tr = np.concatenate([np.concatenate([blocks[name]['tr'], self.acq_block_tr[:-1], [wait_time]]) for name, wait_time in zip(self.prep_order, self.waittimes)])

    def calc_crlb(self, T1, T2, M0, TE, weightingmatrix=None):
        V = calculate_crlb_sc_epg(T1, T2, M0, self.fa, self.tr, TE, return_crlb_matrix=True, weightingmatrix=weightingmatrix)
        self.crlb_T1 = V[0, 0].item()
        self.crlb_T2 = V[1, 1].item()
        self.crlb_M0 = V[2, 2].item()
        self.crlb = self.crlb_T1 + self.crlb_T2 + self.crlb_M0
        

