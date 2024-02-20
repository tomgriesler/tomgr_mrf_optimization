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


def throt(alpha, phi):

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

    