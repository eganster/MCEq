"""The module contains the analytic expressions for the decay distributions of 
neutrons, pions, kaons and muons (including helicity dependence). 

The functions are extracted from a non-public code PriNCe for UHECR propagation.
The documentation of the code and a possible reference is https://arxiv.org/abs/1901.03338 

Note::

    The module is under development and is not yet available.

Authors::

    Anatoli Fedynitch (DESY)
    Jonas Heinze (DESY)

"""

import numpy as np
from MCEq.misc import info
from MCEq.particlemanager import _pdata

def get_particle_channels(mo, mo_energy, da_energy):
    """
    Loops over a all daughers for a given mother and generates
    a list of redistribution matrices on the grid:
     np.outer( da_energy , 1 / mo_energy )
    
    Args:
      mo (int): id of the mother particle
      mo_energy (float): energy grid of the mother particle
      da_energy (float): energy grid of the daughter particle (same for all daughters)
    Returns:
      list of np.array: list of redistribution functions on on xgrid 
    """
    # info(10, 'Generating decay redistribution for', mo, da)
    dbentry = _pdata[mo]
    x_grid = np.outer(da_energy, (1 / mo_energy))

    redist = {}
    for branching, daughters in _pdata.decay_channels(mo):
        for da in daughters:
            # daughter is a nucleus, we have lorentz factor conservation
            if da > 99:
                res = np.zeros(x_grid.shape)
                res[x_grid == 1.] = 1.
            else:
                res = get_decay_matrix(mo, da, x_grid)
            redist[da] = branching * res

    return x_grid, redist


def get_decay_matrix(mo, da, x_grid):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): index of the mother
      da (int): index of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result
                      (If x is a 2D matrix only the last column is computed
                      and then repeated over the matrix assuming that the 
                      main diagonal is always x = 1)
    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    info(10, 'Generating decay redistribution for', mo, da)

    # pi+ to numu or pi- to nummubar
    if abs(mo) == 211 and abs(da) == 14:
        return pion_to_numu(x_grid)

    # pi+ to mu+ or pi- to mu-
    elif abs(mo) == 211 and abs(da) in [13, 7013, 7113, 7213, 7313]:
        # (any helicity)
        if da in [7, 10]:
            return pion_to_muon(x_grid)
        # left handed, hel = -1
        elif da in [5, 8]:
            return pion_to_muon(x_grid) * prob_muon_hel(x_grid, -1.)
        # right handed, hel = 1
        elif da in [6, 9]:
            return pion_to_muon(x_grid) * prob_muon_hel(x_grid, 1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # muon to neutrino
    elif mo in [5, 6, 7, 8, 9, 10] and da in [11, 12, 13, 14]:
        # translating muon ids to helicity
        muon_hel = {
            5: 1.,
            6: -1.,
            7: 0.,
            8: 1.,
            9: -1.,
            10: 0.,
        }
        hel = muon_hel[mo]
        # muon+ to electron neutrino
        if mo in [5, 6, 7] and da in [11]:
            return muonplus_to_nue(x_grid, hel)
        # muon+ to muon anti-neutrino
        elif mo in [5, 6, 7] and da in [14]:
            return muonplus_to_numubar(x_grid, hel)
        # muon- to elec anti-neutrino
        elif mo in [8, 9, 10] and da in [12]:
            return muonplus_to_nue(x_grid, -1 * hel)
        # muon- to muon neutrino
        elif mo in [8, 9, 10] and da in [13]:
            return muonplus_to_numubar(x_grid, -1 * hel)
    else:
        info(
            5,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        return np.zeros(x_grid.shape)


def get_decay_matrix_bin_average(mo, da, x_lower, x_upper):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): index of the mother
      da (int): index of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result

    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    # TODO: Some of the distribution are not averaged yet.
    # The error is small for smooth distributions though
    info(10, 'Generating decay redistribution for', mo, da)

    x_grid = (x_upper + x_lower) / 2

    # remember shape, but only calculate for last column, as x repeats in each column
    shape = x_grid.shape

    if len(shape) == 2:
        x_grid = x_grid[:, -1]
        x_upper = x_upper[:, -1]
        x_lower = x_lower[:, -1]

    # pi+ to numu or pi- to nummubar
    if mo in [211, -211] and da in [14, -14]:
        result = pion_to_numu_avg(x_lower, x_upper)

    # pi+ to mu+ or pi- to mu-
    # TODO: The helicity distr need to be averaged analyticaly
    elif mo in [211, -211] and da in [5, 6, 7, 8, 9, 10]:
        # (any helicity)
        if da in [7, 10]:
            result = pion_to_muon_avg(x_lower, x_upper)
        # left handed, hel = -1
        elif da in [5, 8]:
            result = pion_to_muon_avg(x_lower, x_upper) * prob_muon_hel(
                x_grid, -1.)
        # right handed, hel = 1
        elif da in [6, 9]:
            result = pion_to_muon_avg(x_lower, x_upper) * prob_muon_hel(
                x_grid, 1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # muon to neutrino
    # TODO: The following distr need to be averaged analyticaly
    elif mo in [5, 6, 7, 8, 9, 10] and da in [11, 12, 13, 14]:
        # translating muon ids to helicity
        muon_hel = {
            5: 1.,
            6: -1.,
            7: 0.,
            8: 1.,
            9: -1.,
            10: 0.,
        }
        hel = muon_hel[mo]
        # muon+ to electron neutrino
        if mo in [5, 6, 7] and da in [11]:
            result = muonplus_to_nue(x_grid, hel)
        # muon+ to muon anti-neutrino
        elif mo in [5, 6, 7] and da in [14]:
            result = muonplus_to_numubar(x_grid, hel)
        # muon- to elec anti-neutrino
        elif mo in [8, 9, 10] and da in [12]:
            result = muonplus_to_nue(x_grid, -1 * hel)
        # muon- to muon neutrino
        elif mo in [8, 9, 10] and da in [13]:
            result = muonplus_to_numubar(x_grid, -1 * hel)

    # neutrinos from beta decays
    # TODO: The following beta decay to neutrino distr need to be averaged analyticaly
    # TODO: Also the angular averaging is done numerically still
    # beta-
    elif mo > 99 and da == 11:
        info(10, 'nu_e from beta+ decay', mo, mo - 1, da)
        result = nu_from_beta_decay(x_grid, mo, mo - 1)
    # beta+
    elif mo > 99 and da == 12:
        info(10, 'nubar_e from beta- decay', mo, mo + 1, da)
        result = nu_from_beta_decay(x_grid, mo, mo + 1)
    # neutron
    elif mo > 99 and 99 < da < 200:
        info(10, 'beta decay boost conservation', mo, da)
        result = boost_conservation_avg(x_lower, x_upper)
    else:
        info(
            5,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        result = np.zeros(x_grid.shape)

    # now fill this into diagonals of matrix
    if len(shape) == 2:
        #'filling matrix'
        res_mat = np.zeros(shape)
        for idx, val in enumerate(result[::-1]):
            np.fill_diagonal(res_mat[:, idx:], val)
        result = res_mat

    return result


def pion_to_numu(x):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']
    r = m_muon**2 / m_pion**2
    xmin = 0.
    xmax = 1 - r

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_numu_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception('different grids for xmin, xmax provided')

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']
    r = m_muon**2 / m_pion**2
    xmin = 0.
    xmax = 1 - r

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = 1 / (1 - r)

    return res


def pion_to_muon(x):
    """
    Energy distribution of a muon from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    m_muon = _pdata.mass(13)
    m_pion = _pdata.mass(211)
    r = m_muon**2 / m_pion**2
    xmin = r
    xmax = 1.

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_muon_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception('different grids for xmin, xmax provided')

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = _pdata.mass(13)
    m_pion = _pdata.mass(211)
    r = m_muon**2 / m_pion**2
    xmin = r
    xmax = 1.

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = 1 / (1 - r)

    return res


def prob_muon_hel(x, h):
    """
    Probability for muon+ from pion+ decay to have helicity h
    the result is only valid for x > r

    Args:
      h (int): helicity +/- 1
    Returns:
      float: probability for this helicity
    """

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']

    r = m_muon**2 / m_pion**2

    #helicity expectation value
    hel = 2 * r / (1 - r) / x - (1 + r) / (1 - r)

    res = np.zeros(x.shape)
    cond = np.where(np.logical_and(x > r, x <= 1))
    res[cond] = (1 + hel * h) / 2  #this result is only correct for x > r
    return res


def muonplus_to_numubar(x, h):
    """
    Energy distribution of a numu_bar from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4. / 3., -3., 0., 5. / 3.])
    p2 = np.poly1d([-8. / 3., 3., 0., -1. / 3.])

    res = np.zeros(x.shape)
    cond = x <= 1.
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res


def muonplus_to_nue(x, h):
    """
    Energy distribution of a n from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4., -6., 0., 2.])
    p2 = np.poly1d([-8., 18., -12., 2.])

    res = np.zeros(x.shape)
    cond = x <= 1.
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res