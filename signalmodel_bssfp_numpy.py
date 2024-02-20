import numpy as np


def xrot(alpha):

    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    mat = np.array([
        [1, 0, 0],
        [0, cosalpha, -sinalpha],
        [0, sinalpha, cosalpha]
    ])

    return mat


def zrot(alpha):

    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    mat = np.array([
        [cosalpha, -sinalpha, 0],
        [sinalpha, cosalpha, 0],
        [0, 0, 1]
    ])

    return mat


def q(alpha, phi):

    Rz = zrot(-phi)
    Rx = xrot(alpha)

    mat = np.linalg.inv(Rz) @ Rx @ Rz

    return mat


def freeprecess(t1, t2, t, df):
    """
    df in Hz
    t, t1, t2 in ms
    """

    alpha = 2*np.pi * df * t/1000
    E1 = np.exp(-t/t1)
    E2 = np.exp(-t/t2)

    Afp = np.diag([E2, E2, E1]) @ zrot(alpha)

    Bfp = np.array([[0], [0], [1-E1]])

    return Afp, Bfp


def inversion(inv_eff):

    mat = np.diag([0, 0, -inv_eff])

    return mat


def t2prep(t2, t2te):

    mat = np.diag([0, 0, np.exp(-t2te/t2)])

    return mat


def t1rhoprep(t1rho, tsl):

    mat = np.diag([0, 0, -np.exp(-tsl/t1rho)])

    return mat


def calculate_signal_bssfp(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, tr_offset, te, inv_eff=0.95, delta_B1=1., df=0, t1rho=None):

    n_ex = beats * shots

    Afp_te, Bfp_te = freeprecess(t1, t2, te, df)    
    inv = inversion(inv_eff)

    m = np.vstack((0., 0., m0))

    signal = np.empty(n_ex, dtype=complex)

    for ii in range(beats):

        if prep[ii] ==1:
            Afp_ti, Bfp_ti = freeprecess(t1, t2, ti[ii], df)
            m = Afp_ti @ inv @ m + m0 * Bfp_ti

        elif prep[ii] == 2:
            m = t2prep(t2, t2te[ii]) @ m

        elif prep[ii] == 3:
            m = t1rhoprep(t1rho, t2te[ii]) @ m

        for jj in range(shots):

            n = ii*shots+jj

            q_n = q(delta_B1 * np.deg2rad(fa[n]), np.deg2rad(ph[n]))

            m = Afp_te @ q_n @ m + m0 * Bfp_te

            signal[n] = (m[0] + 1j*m[1]) * np.exp(-1j*np.deg2rad(ph[n]))

            Afp_tr, Bfp_tr = freeprecess(t1, t2, tr_offset+tr[n]*1e-3-te, df)
            m = Afp_tr @ m + m0*Bfp_tr

    return signal