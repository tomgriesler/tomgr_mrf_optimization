import numpy as np

dtype = np.float64

def q(alpha, phi=np.pi/2):
    '''
    Basically taken from Rad229.
    '''
    mat = np.array([
        [(np.cos(alpha/2.))**2., np.exp(2.*1j*phi)*(np.sin(alpha/2.))**2., -1j*np.exp(1j*phi)*np.sin(alpha)],
        [np.exp(-2.*1j*phi)*(np.sin(alpha/2.))**2., (np.cos(alpha/2.))**2., 1j*np.exp(-1j*phi)*np.sin(alpha)],
        [-1j/2.*np.exp(-1j*phi)*np.sin(alpha), 1j/2.*np.exp(1j*phi)*np.sin(alpha), np.cos(alpha)]
    ])

    return mat


def r(T1, T2, t):

    E1 = np.exp(-t/T1)
    E2 = np.exp(-t/T2)

    mat = np.array([
        [E2, 0, 0],
        [0, E2, 0],
        [0, 0, E1]
    ])

    return mat


def dr_dT1(T1, t):
    
    mat = t/T1**2 * np.exp(-t/T1) * np.array([[0, 0, 0],
                                              [0, 0, 0],
                                              [0, 0, 1]])
    
    return mat


def dr_dT2(T2, t):
    
    mat = t/T2**2 * np.exp(-t/T2) * np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]])
    
    return mat


def b(T1, t):

    return (1-np.exp(-t/T1))


def db_dT1(T1, t):

    return (-t/T1**2 * np.exp(-t/T1))


def epg_grad(Omega):

    Omega = np.hstack((Omega, np.zeros((3, 1))))

    Omega[0, 1:] = Omega[0, :-1]
    Omega[1, :-1] = Omega[1, 1:]
    Omega[1, -1] = 0
    Omega[0, 0] = np.conj(Omega[1,0])

    return Omega


def calculate_signal_abdominal(T1, T2, M0, acq_block_fa, acq_block_tr, PREP, TI, T2TE, waittimes, TE, inversion_efficiency=0.95, delta_B1=1, phase=np.pi/2):

    acq_block_fa *= delta_B1

    etl = len(PREP) * len(acq_block_fa)

    R_TE = r(T1, T2, TE)

    Omega = np.vstack((0., 0., M0))

    signal = np.empty(etl, dtype=np.complex64)
    count = 0

    for prep, ti, t2te, waittime in zip(PREP, TI, T2TE, waittimes):

        if prep == 1:
            Omega[:2, :] = 0.
            Omega[2, :] *= -inversion_efficiency
            Omega = r(T1, T2, ti) @ Omega
            Omega[2, 0] += M0 * b(T1, ti)

        elif prep == 2:
            Omega[:2, :] = 0.
            Omega[2, :] *= np.exp(-t2te/T2)

        for fa, tr in zip(acq_block_fa, acq_block_tr):

            Q = q(np.deg2rad(fa), phase)

            signal[count] = (R_TE @ Q @ Omega)[0, 0] * np.exp(-1j*phase)

            Omega = epg_grad(r(T1, T2, tr) @ Q @ Omega)
            Omega[2, 0] += M0 * b(T1, tr)

            count += 1

        Omega = r(T1, T2, waittime) @ Omega
        Omega[2, 0] += M0 * b(T1, waittime)

    return signal


def calculate_crlb_abdominal(T1, T2, M0, acq_block_fa, acq_block_tr, PREP, TI, T2TE, waittimes, TE, inversion_efficiency=0.95, delta_B1=1, sigma=1, phase=np.pi/2):

    acq_block_fa *= delta_B1

    R_TE = r(T1, T2, TE)
    dR_TE_dT1 = dr_dT1(T1, TE)
    dR_TE_dT2 = dr_dT2(T2, TE)    

    Omega = np.vstack((0., 0., M0))
    dOmega_dT1 = np.zeros((3, 1))
    dOmega_dT2 = np.zeros((3, 1))
    dOmega_dM0 = np.vstack((0., 0., 1.))

    I = np.zeros((3, 3))

    for prep, ti, t2te, waittime in zip(PREP, TI, T2TE, waittimes): 

        if prep == 1:

            R_TI = r(T1, T2, ti)

            Omega[:2, :] = 0.
            Omega[2, :] *= -inversion_efficiency

            dOmega_dT1[:2, :] = 0.
            dOmega_dT1[2, :] *= -inversion_efficiency
            dOmega_dT1 = dr_dT1(T1, ti) @ Omega + R_TI @ dOmega_dT1
            dOmega_dT1[2, 0] += M0 * db_dT1(T1, ti)

            dOmega_dT2[:2, :] = 0.
            dOmega_dT2[2, :] *= -inversion_efficiency
            dOmega_dT2 = dr_dT2(T2, ti) @ Omega + R_TI @ dOmega_dT2

            dOmega_dM0[:2, :] = 0.
            dOmega_dM0[2, :] *= -inversion_efficiency
            dOmega_dM0 = R_TI @ dOmega_dM0

            Omega = R_TI @ Omega
            Omega[2, 0] += M0 * b(T1, ti)

        elif prep == 2:

            Omega[:2, :] = 0.

            dOmega_dT1[:2, :] = 0
            dOmega_dT1 *= np.exp(-t2te/T2)

            dOmega_dT2[:2, :] = 0
            dOmega_dT2 = np.exp(-t2te/T2) * (t2te/T2**2 * Omega + dOmega_dT2)

            dOmega_dM0[:2, :] = 0
            dOmega_dM0 *= np.exp(-t2te/T2)

            Omega[2, :] *= np.exp(-t2te/T2)

        for fa, tr in zip(acq_block_fa, acq_block_tr): 

            Q = q(np.deg2rad(fa), phase)
            R_TR = r(T1, T2, tr)
            dR_TR_dT1 = dr_dT1(T1, tr)
            dR_TR_dT2 = dr_dT2(T2, tr)
            b_TR = b(T1, tr)
            B_TR = M0 * b_TR
            dB_TR_dT1 = M0 * db_dT1(T1, tr)
            dB_TR_dM0 = b_TR

            # Calculate derivatives of signal 
            dsignal_dT1 = (dR_TE_dT1 @ Q @ Omega + R_TE @ Q @ dOmega_dT1)[0, 0]
            dsignal_dT2 = (dR_TE_dT2 @ Q @ Omega + R_TE @ Q @ dOmega_dT2)[0, 0]
            dsignal_dM0 = (R_TE @ Q @ dOmega_dM0)[0, 0]

            # Calculate Jacobian
            J_n = np.array([
                [np.real(dsignal_dT1), np.real(dsignal_dT2), np.real(dsignal_dM0)],
                [np.imag(dsignal_dT1), np.imag(dsignal_dT2), np.imag(dsignal_dM0)]
            ])

            J_n_t = np.transpose(J_n)

            # Calculate FIM
            I = I + 1/sigma**2 * (J_n_t @ J_n)
            
            # Calculate new state matrix and derivatives
            dOmega_dT1 = epg_grad(dR_TR_dT1 @ Q @ Omega + R_TR @ Q @ dOmega_dT1) 
            dOmega_dT1[2, 0] += dB_TR_dT1

            dOmega_dT2 = epg_grad(dR_TR_dT2 @ Q @ Omega + R_TR @ Q @ dOmega_dT2)

            dOmega_dM0 = epg_grad(R_TR @ Q @ dOmega_dM0)
            dOmega_dM0[2, 0] += dB_TR_dM0

            # Update Magnetization
            Omega = epg_grad(R_TR @ Q @ Omega)
            Omega[2, 0] += B_TR

        Omega = r(T1, T2, waittime) @ Omega
        Omega[2, 0] += M0 * b(T1, waittime)

    V = np.linalg.inv(I)

    return V