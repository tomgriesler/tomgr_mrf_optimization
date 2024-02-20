import numpy as np
import matplotlib.pyplot as plt


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