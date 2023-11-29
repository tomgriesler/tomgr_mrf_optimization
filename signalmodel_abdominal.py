import numpy as np


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


def inversion(inversion_efficiency):

    mat = np.zeros((3, 3))
    mat[2, 2] = -inversion_efficiency

    return mat


def t2prep(t2, t2te):

    mat = np.zeros((3, 3))
    mat[2, 2] = np.exp(-t2te/t2)

    return mat


def dt2prep_dT2(t2, t2te):

    mat = np.zeros((3, 3))
    mat[2, 2] = t2te/t2**2 * np.exp(-t2te/t2)

    return mat


def r(t1, t2, t):

    E1 = np.exp(-t/t1)
    E2 = np.exp(-t/t2)

    mat = np.array([
        [E2, 0, 0],
        [0, E2, 0],
        [0, 0, E1]
    ])

    return mat


def dr_dt1(t1, t):
    
    mat = t/t1**2 * np.exp(-t/t1) * np.array([[0, 0, 0],
                                              [0, 0, 0],
                                              [0, 0, 1]])
    
    return mat


def dr_dt2(t2, t):
    
    mat = t/t2**2 * np.exp(-t/t2) * np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]])
    
    return mat


def b(t1, t):

    return (1-np.exp(-t/t1))


def db_dt1(t1, t):

    return (-t/t1**2 * np.exp(-t/t1))


def epg_grad(omega):

    omega = np.hstack((omega, np.zeros((3, 1))))

    omega[0, 1:] = omega[0, :-1]
    omega[1, :-1] = omega[1, 1:]
    omega[1, -1] = 0
    omega[0, 0] = np.conj(omega[1,0])

    return omega


def calculate_signal(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inversion_efficiency=0.95, delta_B1=1.):

    n_ex = beats * shots

    r_te = r(t1, t2, te)
    b_te = b(t1, te)
    inv = inversion(inversion_efficiency)

    omega = np.vstack((0., 0., m0))

    signal = np.empty(n_ex, dtype=complex)
    
    for ii in range(beats):
        
        if prep[ii] == 1:
            omega = r(t1, t2, ti[ii]) @ inv @ omega
            omega[2, 0] += m0 * b(t1, ti[ii])

        elif prep[ii] == 2: 
            omega = t2prep(t2, t2te[ii]) @ omega

        for jj in range(shots):

            n = ii*shots+jj

            q_n = q(delta_B1*np.deg2rad(fa[n]), np.deg2rad(ph[n]))

            omega = r_te @ q_n @ omega
            omega[2, 0] += m0 * b_te

            signal[n] = omega[0, 0] * np.exp(-1j*np.deg2rad(ph[n]))

            omega = epg_grad(r(t1, t2, tr_offset+tr[n]*1e-3-te) @ omega)
            omega[2, 0] += m0 * b(t1, tr_offset+tr[n]*1e-3-te)

    return signal

def calculate_crlb(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inversion_efficiency=0.95, delta_B1=1.):

    r_te = r(t1, t2, te)
    dr_te_dt1 = dr_dt1(t1, te)
    dr_te_dt2 = dr_dt2(t2, te)    
    inv = inversion(inversion_efficiency)

    omega = np.vstack((0., 0., m0))
    domega_dt1 = np.zeros((3, 1))
    domega_dt2 = np.zeros((3, 1))
    domega_dm0 = np.vstack((0., 0., 1.))

    fim = np.zeros((3, 3))

    for ii in range(beats): 

        if prep[ii] == 1:

            r_ti = r(t1, t2, ti[ii])

            domega_dt1 = dr_dt1(t1, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt1
            domega_dt1[2, 0] += m0 * db_dt1(t1, ti[ii])

            domega_dt2 = dr_dt2(t2, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt2

            domega_dm0 = r_ti @ inv @ domega_dm0

            omega = r_ti @ inv @ omega
            omega[2, 0] += m0 * b(t1, ti[ii])

        elif prep[ii] == 2:

            t2prep_ii = t2prep(t2, t2te[ii])
            dt2prep_ii_dT2 = dt2prep_dT2(t2, t2te[ii])

            domega_dt1 = t2prep_ii @ domega_dt1

            domega_dt2 = dt2prep_ii_dT2 @ omega + t2prep_ii @ domega_dt2

            domega_dm0 = t2prep_ii @ domega_dm0

            omega = t2prep_ii @ omega

        for jj in range(shots):

            n = ii*shots + jj

            q_n = q(delta_B1*np.deg2rad(fa[n]), np.deg2rad(ph[n]))

            r_tr = r(t1, t2, tr_offset+tr[n]*1e-3)
            dr_tr_dt1 = dr_dt1(t1, tr_offset+tr[n]*1e-3)
            dr_tr_dt2 = dr_dt2(t2, tr_offset+tr[n]*1e-3)
            b_tr = m0 * b(t1, tr_offset+tr[n]*1e-3)
            db_tr_dt1 = m0 * db_dt1(t1, tr_offset+tr[n]*1e-3)
            db_tr_dm0 = b_tr/m0

            # Calculate derivatives of signal 
            dsignal_dt1 = (dr_te_dt1 @ q_n @ omega + r_te @ q_n @ domega_dt1)[0, 0]
            dsignal_dt2 = (dr_te_dt2 @ q_n @ omega + r_te @ q_n @ domega_dt2)[0, 0]
            dsignal_dm0 = (r_te @ q_n @ domega_dm0)[0, 0]

            # Calculate Jacobian
            J_n = np.array([
                [np.real(dsignal_dt1), np.real(dsignal_dt2), np.real(dsignal_dm0)],
                [np.imag(dsignal_dt1), np.imag(dsignal_dt2), np.imag(dsignal_dm0)]
            ])

            J_n_t = np.transpose(J_n)

            # Calculate FIM
            fim = fim + (J_n_t @ J_n)
            
            # Calculate new state matrix and derivatives
            domega_dt1 = epg_grad(dr_tr_dt1 @ q_n @ omega + r_tr @ q_n @ domega_dt1) 
            domega_dt1[2, 0] += db_tr_dt1

            domega_dt2 = epg_grad(dr_tr_dt2 @ q_n @ omega + r_tr @ q_n @ domega_dt2)

            domega_dm0 = epg_grad(r_tr @ q_n @ domega_dm0)
            domega_dm0[2, 0] += db_tr_dm0

            # Update Magnetization
            omega = epg_grad(r_tr @ q_n @ omega)
            omega[2, 0] += b_tr

    return np.linalg.inv(fim)


def calculate_crlb_pv(t1, t2, m0, fraction, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inversion_efficiency=0.95, delta_B1=1.):

    r_te_1 = r(t1[0], t2[0], te)
    r_te_2 = r(t1[1], t2[1], te)

    omega_1 = np.vstack((0., 0., fraction*m0))
    omega_2 = np.vstack((0., 0., (1-fraction)*m0))

    domega_1_dfraction = np.vstack((0., 0., m0))
    domega_2_dfraction = np.vstack((0., 0., -m0))

    fim = 0.

    for ii in range(beats):

        if prep[ii] == 1:

            omega_1[:2, :] = 0.
            omega_2[:2, :] = 0.

            omega_1[2, :] *= -inversion_efficiency
            omega_2[2, :] *= -inversion_efficiency



