# -*- coding: utf-8 -*-
"""
:mod:`MCEq.misc` - other useful things
======================================

Some helper functions and plotting features are collected in this module.

- :class:`EdepZFactos` calculates energy-dependent spectrum weighted
  moments (Z-Factors)

"""
from __future__ import print_function

import numpy as np
from mceq_config import config, dbg


def normalize_hadronic_model_name(name):
    """Converts hadronic model name into standard form"""
    return name.translate(None, ".-").upper()


def theta_deg(cos_theta):
    """Converts :math:`\\cos{\\theta}` to :math:`\\theta` in degrees.
    """
    return np.rad2deg(np.arccos(cos_theta))


def theta_rad(theta):
    """Converts :math:`\\theta` from rad to degrees.
    """
    return np.deg2rad(theta)


def print_in_rows(min_dbg_level, str_list, n_cols=8):
    """Prints contents of a list in rows `n_cols`
    entries per row.
    """
    if min_dbg_level > config["debug_level"]:
        return
        
    l = len(str_list)
    n_full_length = int(l / n_cols)
    n_rest = l % n_cols
    print_str = '\n'
    for i in range(n_full_length):
        print_str += ('"{:}", ' * n_cols
                      ).format(*str_list[i * n_cols:(i + 1) * n_cols]) + '\n'
    print_str += ('"{:}", ' * n_rest).format(*str_list[-n_rest:])

    print(print_str.strip()[:-1])

def is_charm_pdgid(pdgid):
    """Returns True if particle ID belongs to a heavy (charm) hadron."""

    return ((abs(pdgid) > 400 and abs(pdgid) < 500)
            or (abs(pdgid) > 4000 and abs(pdgid) < 5000))

def _get_closest(value, in_list):
    """Returns the closes value to 'value' from given list."""

    minindex = np.argmin(np.abs(in_list - value * np.ones(len(in_list))))
    return minindex, in_list[minindex]

class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
    """

    def __init__(self, lower, upper, bins_dec):
        import numpy as np
        self.bins = np.logspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = np.sqrt(self.bins[1:] * self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(
            1, 'Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
                self.bins[0], self.bins[-1], self.grid.size))


class EdepZFactors():
    """Handles calculation of energy dependent Z factors.

    Was not recently checked and results could be wrong."""

    def __init__(self, interaction_model, primary_flux_model):
        from MCEq.data import InteractionYields, HadAirCrossSections
        from particletools.tables import SibyllParticleTable

        self.y = InteractionYields(interaction_model)
        self.cs = HadAirCrossSections(interaction_model)

        self.pm = primary_flux_model
        self.e_bins, self.e_widths = self._get_bins_and_width_from_centers(
            self.y.e_grid)
        self.e_vec = self.y.e_grid
        self.iamod = interaction_model
        self.sibtab = SibyllParticleTable()
        self._gen_integrator()

    def _get_bins_and_width_from_centers(self, vector):
        """Returns bins and bin widths given given bin centers."""

        vector_log = np.log10(vector)
        steps = vector_log[1] - vector_log[0]
        bins_log = vector_log - 0.5 * steps
        bins_log = np.resize(bins_log, vector_log.size + 1)
        bins_log[-1] = vector_log[-1] + 0.5 * steps
        bins = 10**bins_log
        widths = bins[1:] - bins[:-1]
        return bins, widths

    def get_zfactor(self, proj, sec_hadr, logx=False, use_cs=True):
        proj_cs_vec = self.cs.get_cs(proj)
        nuc_flux = self.pm.tot_nucleon_flux(self.e_vec)
        zfac = np.zeros(self.y.dim)
        if self.y.is_yield(proj, sec_hadr):
            info(1, "calculating zfactor Z({0},{1})".format(proj, sec_hadr))
            y_mat = self.y.get_y_matrix(proj, sec_hadr)

            self.calculate_zfac(self.e_vec, self.e_widths, nuc_flux,
                                proj_cs_vec, y_mat, zfac, use_cs)

        if logx:
            return np.log10(self.e_vec), zfac
        return self.e_vec, zfac

    def _gen_integrator(self):
        try:
            from numba import jit, double, boolean, void

            @jit(
                void(double[:], double[:], double[:], double[:], double[:, :],
                     double[:], boolean),
                target='cpu')
            def calculate_zfac(e_vec, e_widths, nuc_flux, proj_cs, y, zfac,
                               use_cs):
                for h, E_h in enumerate(e_vec):
                    for k in range(len(e_vec)):
                        E_k = e_vec[k]
                        # dE_k = e_widths[k]
                        if E_k < E_h:
                            continue
                        csfac = proj_cs[k] / proj_cs[h] if use_cs else 1.

                        zfac[h] += nuc_flux[k] / nuc_flux[h] * csfac * \
                            y[:, k][h]  # * dE_k
        except ImportError:
            raise Exception("Warning! Numba not in PYTHONPATH. ZFactor " +
                            "calculation won't work.")

        self.calculate_zfac = calculate_zfac


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
    An empty string is returned if skipped levels exceed stack height.abs

    From https://gist.github.com/techtonik/2151727
    """
    import inspect

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ''

    parentframe = stack[start][0]

    name = []

    if config["print_module"]:
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        if module:
            name.append(module.__name__ + '.')

    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # there seems to be no way to detect static method call - it will
        # be just a function call

        name.append(parentframe.f_locals['self'].__class__.__name__ + '::')

    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename + '(): ')  # function or a method
    else:
        name.append(': ')  # If called from module scope

    del parentframe
    return "".join(name)


def info(min_dbg_level, *message, **kwargs):
    """Print to console if `min_debug_level <= config["debug_level"]`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
        condition (bool): Print only if condition is True
        blank_caller (bool): blank the caller name (for multiline output)
        no_caller (bool): don't print the name of the caller

    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """

    condition = kwargs.pop('condition', True)
    blank_caller = kwargs.pop('blank_caller', False)
    no_caller = kwargs.pop('no_caller', False)

    if condition and min_dbg_level <= config["debug_level"]:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ''
        if blank_caller: cname = len(cname) * ' '
        print(cname + " ".join(message))