import numpy as np
from numpy.linalg import LinAlgError


def q(alpha, phi=np.pi/2):

    mat = np.array([
        [(np.cos(alpha/2.))**2., np.exp(2.*1j*phi)*(np.sin(alpha/2.))**2., -1j*np.exp(1j*phi)*np.sin(alpha)],
        [np.exp(-2.*1j*phi)*(np.sin(alpha/2.))**2., (np.cos(alpha/2.))**2., 1j*np.exp(-1j*phi)*np.sin(alpha)],
        [-1j/2.*np.exp(-1j*phi)*np.sin(alpha), 1j/2.*np.exp(1j*phi)*np.sin(alpha), np.cos(alpha)]
    ])

    return mat


def inversion(inv_eff):

    mat = np.diag([0, 0, -inv_eff])

    return mat


def t2prep(t2, t2te):

    mat = np.diag([0, 0, np.exp(-t2te/t2)])

    return mat


def dt2prep_dt2(t2, t2te):

    mat = np.diag([0, 0, t2te/t2**2 * np.exp(-t2te/t2)])

    return mat


def t1rhoprep(t1rho, tsl):

    mat = np.diag([0, 0, -np.exp(-tsl/t1rho)])

    return mat


def dt1rhoprep_dt1rho(t1rho, tsl):

    mat = np.diag([0, 0, -tsl/t1rho**2 * np.exp(-tsl/t1rho)])

    return mat


def r(t1, t2, t):

    E1 = np.exp(-t/t1)
    E2 = np.exp(-t/t2)

    mat = np.diag([E2, E2, E1])

    return mat


def dr_dt1(t1, t):

    mat = np.diag([0, 0, t/t1**2 * np.exp(-t/t1)])
    
    return mat


def dr_dt2(t2, t):
    
    mat = t/t2**2 * np.exp(-t/t2) * np.diag([1, 1, 0])
    
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


def calculate_signal(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff=0.95, delta_B1=1., t1rho=None, tsl=None):

    n_ex = beats * shots

    r_te = r(t1, t2, te)
    b_te = b(t1, te)
    inv = inversion(inv_eff)

    omega = np.vstack((0., 0., m0))

    signal = np.empty(n_ex, dtype=complex)
    
    for ii in range(beats):
        
        if prep[ii] == 1:
            omega = r(t1, t2, ti[ii]) @ inv @ omega
            omega[2, 0] += m0 * b(t1, ti[ii])

        elif prep[ii] == 2: 
            omega = t2prep(t2, t2te[ii]) @ omega

        elif prep[ii] == 3: 
            omega = t1rhoprep(t1rho, tsl[ii]) @ omega

        for jj in range(shots):

            n = ii*shots+jj

            q_n = q(delta_B1*np.deg2rad(fa[n]), np.deg2rad(ph[n]))

            omega = r_te @ q_n @ omega
            omega[2, 0] += m0 * b_te

            signal[n] = omega[0, 0] * np.exp(-1j*np.deg2rad(ph[n]))

            omega = epg_grad(r(t1, t2, tr_offset+tr[n]*1e-3-te) @ omega)
            omega[2, 0] += m0 * b(t1, tr_offset+tr[n]*1e-3-te)

    return signal


def calculate_crlb(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff=0.95, delta_B1=1., t1rho=None, tsl=None):

    r_te = r(t1, t2, te)
    dr_te_dt1 = dr_dt1(t1, te)
    dr_te_dt2 = dr_dt2(t2, te)    
    inv = inversion(inv_eff)

    omega = np.vstack((0., 0., m0))
    domega_dt1 = np.zeros((3, 1))
    domega_dt2 = np.zeros((3, 1))
    domega_dm0 = np.vstack((0., 0., 1.))
    domega_dt1rho = np.zeros((3, 1)) if t1rho is not None else None 

    fim = np.zeros((3, 3)) if t1rho is None else np.zeros((4, 4))

    for ii in range(beats): 

        if prep[ii] == 1:

            r_ti = r(t1, t2, ti[ii])

            domega_dt1 = dr_dt1(t1, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt1
            domega_dt1[2, 0] += m0 * db_dt1(t1, ti[ii])

            domega_dt2 = dr_dt2(t2, ti[ii]) @ inv @ omega + r_ti @ inv @ domega_dt2

            domega_dm0 = r_ti @ inv @ domega_dm0

            domega_dt1rho = r_ti @ inv @ domega_dt1rho if t1rho is not None else None

            omega = r_ti @ inv @ omega
            omega[2, 0] += m0 * b(t1, ti[ii])

        elif prep[ii] == 2:

            t2prep_ii = t2prep(t2, t2te[ii])
            dt2prep_ii_dt2 = dt2prep_dt2(t2, t2te[ii])

            domega_dt1 = t2prep_ii @ domega_dt1

            domega_dt2 = dt2prep_ii_dt2 @ omega + t2prep_ii @ domega_dt2

            domega_dm0 = t2prep_ii @ domega_dm0

            domega_dt1rho = t2prep_ii @ domega_dt1rho if t1rho is not None else None

            omega = t2prep_ii @ omega

        elif prep[ii] == 3:
            
            t1rhoprep_ii = t1rhoprep(t1rho, tsl[ii])
            dt1rhoprep_ii_dt1rho = dt1rhoprep_dt1rho(t1rho, tsl[ii])

            domega_dt1 = t1rhoprep_ii @ domega_dt1

            domega_dt2 = t1rhoprep_ii @ domega_dt2

            domega_dm0 = t1rhoprep_ii @ domega_dm0

            domega_dt1rho = dt1rhoprep_ii_dt1rho @ omega + t1rhoprep_ii @ domega_dt1rho

            omega = t1rhoprep_ii @ omega

        for jj in range(shots):

            n = ii*shots + jj

            q_n = q(delta_B1*np.deg2rad(fa[n]), np.deg2rad(ph[n]))

            # Calculate derivatives of signal at TE
            dsignal_dt1 = (dr_te_dt1 @ q_n @ omega + r_te @ q_n @ domega_dt1)[0, 0]
            dsignal_dt2 = (dr_te_dt2 @ q_n @ omega + r_te @ q_n @ domega_dt2)[0, 0]
            dsignal_dm0 = (r_te @ q_n @ domega_dm0)[0, 0]

            # Calculate Jacobian
            if t1rho is None:
                J_n = np.array([
                    [np.real(dsignal_dt1), np.real(dsignal_dt2), np.real(dsignal_dm0)],
                    [np.imag(dsignal_dt1), np.imag(dsignal_dt2), np.imag(dsignal_dm0)]
                ])
            else:
                dsignal_dt1rho = (r_te @ q_n @ domega_dt1rho)[0, 0]
                J_n = np.array([
                    [np.real(dsignal_dt1), np.real(dsignal_dt2), np.real(dsignal_dm0), np.real(dsignal_dt1rho)],
                    [np.imag(dsignal_dt1), np.imag(dsignal_dt2), np.imag(dsignal_dm0), np.imag(dsignal_dt1rho)]
                ])

            J_n_t = np.transpose(J_n)

            # Calculate FIM
            fim = fim + (J_n_t @ J_n)

            # Calculate new state matrix and derivatives
            r_tr = r(t1, t2, tr_offset+tr[n]*1e-3)
            dr_tr_dt1 = dr_dt1(t1, tr_offset+tr[n]*1e-3)
            dr_tr_dt2 = dr_dt2(t2, tr_offset+tr[n]*1e-3)
            b_tr = m0 * b(t1, tr_offset+tr[n]*1e-3)
            db_tr_dt1 = m0 * db_dt1(t1, tr_offset+tr[n]*1e-3)
            db_tr_dm0 = b_tr/m0
            
            domega_dt1 = epg_grad(dr_tr_dt1 @ q_n @ omega + r_tr @ q_n @ domega_dt1) 
            domega_dt1[2, 0] += db_tr_dt1

            domega_dt2 = epg_grad(dr_tr_dt2 @ q_n @ omega + r_tr @ q_n @ domega_dt2)

            domega_dm0 = epg_grad(r_tr @ q_n @ domega_dm0)
            domega_dm0[2, 0] += db_tr_dm0

            domega_dt1rho = epg_grad(r_tr @ q_n @ domega_dt1rho) if t1rho is not None else None

            # Update Magnetization
            omega = epg_grad(r_tr @ q_n @ omega)
            omega[2, 0] += b_tr

    try:
        v = np.linalg.inv(fim)
    except LinAlgError:
        v = np.zeros_like(fim)
        np.fill_diagonal(v, np.inf)
        
    return v


def calculate_crlb_pv(t1, t2, m0, fraction, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff=0.95, delta_B1=1.):

    r_te_1 = r(t1[0], t2[0], te)
    r_te_2 = r(t1[1], t2[1], te)

    inv = inversion(inv_eff)

    omega_1 = np.vstack((0., 0., fraction*m0))
    omega_2 = np.vstack((0., 0., (1-fraction)*m0))

    domega_1_dfraction = np.vstack((0., 0., m0))
    domega_2_dfraction = np.vstack((0., 0., -m0))

    fim = 0.

    for ii in range(beats):

        if prep[ii] == 1:

            r_ti_1 = r(t1[0], t2[0], ti[ii])
            r_ti_2 = r(t1[1], t2[1], ti[ii])

            domega_1_dfraction = r_ti_1 @ inv @ domega_1_dfraction
            domega_2_dfraction = r_ti_2 @ inv @ domega_2_dfraction
            
            omega_1 = r_ti_1 @ inv @ omega_1
            omega_2 = r_ti_2 @ inv @ omega_2

        elif prep[ii] == 2:

            t2prep_ii_1 = t2prep(t2[0], t2te[ii])
            t2prep_ii_2 = t2prep(t2[1], t2te[ii])

            domega_1_dfraction = t2prep_ii_1 @ domega_1_dfraction
            domega_2_dfraction = t2prep_ii_2 @ domega_2_dfraction

            omega_1 = t2prep_ii_1 @ omega_1
            omega_2 = t2prep_ii_2 @ omega_2

        for jj in range(shots):

            n = ii*shots + jj

            q_n = q(delta_B1*np.deg2rad(fa[n]), np.deg2rad(ph[n]))
            
            # Calculate derivative of signal at TE
            dsignal_dfraction = (r_te_1 @ q_n @ omega_1)[0, 0] + (r_te_2 @ q_n @ omega_2)[0, 0]

            # Calculate FIM
            fim += np.abs(dsignal_dfraction)**2

            # Calculate new state matrix and derivatives
            r_tr_1 = r(t1[0], t2[0], tr_offset+tr[n]*1e-3)
            r_tr_2 = r(t1[1], t2[1], tr_offset+tr[n]*1e-3)
            b_tr_1 = fraction * m0 * b(t1[0], tr_offset+tr[n]*1e-3)
            b_tr_2 = (1-fraction) * m0 * b(t1[1], tr_offset+tr[n]*1e-3)
            db_tr_1_dfraction = b_tr_1/fraction
            db_tr_2_dfraction = -b_tr_2/(1-fraction)

            domega_1_dfraction = epg_grad(r_tr_1 @ q_n @ domega_1_dfraction)
            domega_1_dfraction[2, 0] += db_tr_1_dfraction
            domega_2_dfraction = epg_grad(r_tr_2 @ q_n @ domega_2_dfraction)
            domega_2_dfraction[2, 0] += db_tr_2_dfraction

            # Update magnetization
            omega_1 = epg_grad(r_tr_1 @ q_n @ omega_1)
            omega_1[2, 0] += b_tr_1
            omega_2 = epg_grad(r_tr_2 @ q_n @ omega_2)
            omega_2[2, 0] += b_tr_2

    return 1/fim


def calculate_orthogonality(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff=0.95, delta_B1=1., t1rho=None, tsl=None):

    n_components = len(t1)

    s = np.zeros((n_components, beats*shots), dtype=np.complex128)

    for ii in range(n_components): 
        s[ii] = calculate_signal(t1[ii], t2[ii], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff, delta_B1, t1rho[ii], tsl)

    s = s/np.linalg.norm(s, axis=1, keepdims=True)

    return np.linalg.norm(np.eye(n_components) - s@np.conj(s).T)**2


def calculate_crlb_orthogonality_combined(t1, t2, t1rho, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tsl, tr_offset, te, inv_eff=0.95, delta_B1=1.):

    crlb_1 = np.sqrt(calculate_crlb(t1[0], t2[0], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff, delta_B1, t1rho[0], tsl)[-1, -1] / t1rho[0])

    crlb_2 = np.sqrt(calculate_crlb(t1[1], t2[1], m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff, delta_B1, t1rho[0], tsl)[-1, -1] / t1rho[1])

    orth = calculate_orthogonality(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff, delta_B1, t1rho, tsl)

    return 1 / (crlb_1 + crlb_2) / orth