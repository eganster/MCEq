# -*- coding: utf-8 -*-
"""
:mod:`MCEq.data` --- data management
====================================

This module includes code for bookkeeping, interfacing and
validating data structures:

- :class:`InteractionYields` manages particle interactions, obtained
  from sampling of various interaction models
- :class:`DecayYields` manages particle decays, obtained from
  sampling PYTHIA8 Monte Carlo
- :class:`HadAirCrossSections` keeps information about the inelastic,
  cross-section of hadrons with air. Typically obtained from Monte Carlo.
"""

import numpy as np
from mceq_config import config
from misc import normalize_hadronic_model_name, is_charm_pdgid, info

class InteractionYields(object):
    """Class for managing the dictionary of interaction yield matrices.

    The class unpickles a dictionary, which contains the energy grid
    and :math:`x` spectra, sampled from hadronic interaction models.



    A list of available interaction model keys can be printed by::

        $ print yield_obj

    Args:
      interaction_model (str): name of the interaction model
      charm_model (str, optional): name of the charm model

    """

    def __init__(self, interaction_model, charm_model=None):
        from collections import defaultdict

        #: (str) InterAction Model name
        self.iam = None
        #: (str) charm model name
        self.charm_model = None
        #: (numpy.array) energy grid bin centers
        self.e_grid = None
        #: (numpy.array) energy grid bin endges
        self.e_bins = None
        #: (numpy.array) energy grid bin widths
        self.widths = None
        #: (int) dimension of grid
        self.dim = 0
        #: (tuple) selection of a band of coeffictients (in xf)
        self.band = None
        #: (tuple) modified particle combination for error prop.
        self.mod_pprod = defaultdict(lambda: {})
        #: (numpy.array) Matrix of x_lab values
        self.xmat = None
        #: (list) List of particles supported by interaction model
        self.particle_list = []

        # If parameters are provided during object creation,
        # load the tables during object creation.
        interaction_model = normalize_hadronic_model_name(interaction_model)
        if interaction_model != None:
            self._load(interaction_model)
        else:
            info(2, 'Loading SIBYLL 2.1 by default.')
            self._load('SIBYLL2.1')

        if charm_model and interaction_model:
            self._inject_custom_charm_model(charm_model)

    def _load(self, interaction_model):
        """Un-pickles the yields dictionary using the path specified as
        ``yield_fname`` in :mod:`mceq_config`.

        Class attributes :attr:`e_grid`, :attr:`e_bins`, :attr:`widths`,
        :attr:`dim` are set here.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join, isfile
        from MCEq.data_utils import convert_to_compact, extend_to_low_energies
        info(5, interaction_model)

        # Remove dashes and points in the name
        iamstr = normalize_hadronic_model_name(interaction_model)

        fname = join(config['data_dir'], iamstr + '_yields.bz2')

        if config['compact_mode'] and config["low_energy_extension"]["enabled"] \
                and 'DPMJET' not in iamstr:
            fname = fname.replace('.bz2', '_compact_ledpm.bz2')
        elif not config['compact_mode'] and config["low_energy_extension"]["enabled"] \
                and ('DPMJET' not in iamstr):
            fname = fname.replace('.bz2', '_ledpm.bz2')
        elif config['compact_mode']:
            fname = fname.replace('.bz2', '_compact.bz2')

        yield_dict = None
        info(5, 'Looking for', fname)
        if not isfile(fname):
            if config['compact_mode']:
                convert_to_compact(fname)
            elif 'ledpm' in fname:
                extend_to_low_energies(fname=fname)
            else:
                raise Exception('no model file found for' +
                    interaction_model)

        if not isfile(fname.replace('.bz2', '.ppd')):
            self._decompress(fname)

        yield_dict = pickle.load(open(fname.replace('.bz2', '.ppd'), 'rb'))

        self.e_grid = yield_dict.pop('evec')
        self.e_bins = yield_dict.pop('ebins')
        self.widths = yield_dict.pop('widths')
        self.iam = normalize_hadronic_model_name(yield_dict.pop('mname'))
        self.projectiles = yield_dict.pop('projectiles')
        self.secondary_dict = yield_dict.pop('secondary_dict')
        self.nspec = yield_dict.pop('nspec')

        self.yields = yield_dict

        #  = np.diag(self.e_bins[1:] - self.e_bins[:-1])
        self.dim = self.e_grid.size
        self.no_interaction = np.zeros(self.dim**2).reshape(self.dim, self.dim)

        self.charm_model = None

        self._gen_particle_list()

    def _gen_index(self, yield_dict):
        """Generates index of mother-daughter relationships.

        Currently this function is called each time an interaction model
        is set. In future versions this index will be part of the pickled
        dictionary.

        Args:
          yield_dict (dict): dictionary of yields for one interaction model
        """

        info(5, 'Generating index.')

        ptemp = np.unique(zip(*yield_dict.keys())[0])

        # Filter out the non numerical strings from this list
        projectiles = []
        for proj in ptemp:
            try:
                projectiles.append(int(proj))
            except ValueError:
                continue

        e_bins = yield_dict['ebins']
        widths = np.diag(e_bins[1:] - e_bins[:-1])
        e_grid = np.sqrt(e_bins[1:] * e_bins[:-1])

        secondary_dict = {}

        for projectile in projectiles:
            secondary_dict[projectile] = []

        # New dictionary to replace yield_dict
        new_dict = {}

        for key, mat in yield_dict.iteritems():
            try:
                proj, sec = key
            except ValueError:
                info(5, 'Copy additional info', key)
                # Copy additional items to the new dictionary
                new_dict[key] = mat
                continue

            # exclude electrons and photons
            if np.sum(mat) > 0:  # and abs(sec) not in [11, 22]:
                # print sec not in secondary_dict[proj]
                assert(sec not in secondary_dict[proj]), \
                    ("Error in construction of index array: {0} -> {1}".format(proj, sec))
                secondary_dict[proj].append(sec)

                # Multiply by widths (energy bin widths with matrices)
                new_dict[key] = mat.dot(widths)
            else:
                info(5, 'Zero yield matrix for', key)

        new_dict['projectiles'] = projectiles
        new_dict['secondary_dict'] = secondary_dict
        new_dict['nspec'] = len(projectiles)
        new_dict['widths'] = widths
        new_dict['evec'] = e_grid
        new_dict['ebins'] = e_bins

        if 'le_ext' not in new_dict:
            new_dict['le_ext'] = config['low_energy_extension']

        return new_dict

    def _gen_particle_list(self):
        """Saves a list of all particles handled by selected model.
        """

        # Look up all particle species that a supported by selected model
        for p, l in self.secondary_dict.iteritems():
            self.particle_list += [p]
            self.particle_list += l
        self.particle_list = list(set(self.particle_list))

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(
                self.__class__.__name__ +
                '::_decompress():: File {0} not found.'.format(fcompr))

        info(2, 'Decompressing ', fcompr)

        # Generate index of primary secondary relations and
        # multiply with yields
        new_dict = self._gen_index(pickle.load(bz2.BZ2File(fcompr)))

        # Dump the file uncompressed
        info(2, 'Saving to ', fname.replace('.bz2', '.ppd'))
        pickle.dump(
            new_dict, open(fname.replace('.bz2', '.ppd'), 'wb'), protocol=-1)

    def _gen_mod_matrix(self, x_func, *args):
        """Creates modification matrix using an (x,E)-dependent function.

        :math:`x = \\frac{E_{\\rm primary}}{E_{\\rm secondary}}` is the
        fraction of secondary particle energy. ``x_func`` can be an
        arbitrary function modifying the :math:`x_\\text{lab}` distribution.
        Run this method each time you change ``x_func``, or its parameters,
        not each time you change modified particle.
        The ``args`` are passed to the function.

        Args:
          x_func (object): reference to function
          args (tuple): arguments of `x_func`

        Returns:
          (numpy.array): modification matrix
        """

        info(2, 'Generating modification matrix for', x_func.__name__, args)

        # if not config['error_propagation_mode']:
        #     raise Exception(self.__class__.__name__ +
        #             'init_mod_matrix(): enable error ' +
        #             'propagation mode in config and re-initialize MCEqRun.')

        info(5, 'creating xmat')
        if self.xmat is None:
            self.xmat = self.no_interaction
            for eidx in range(self.dim):
                xvec = self.e_grid[:eidx + 1] / self.e_grid[eidx]
                self.xmat[:eidx + 1, eidx] = xvec

        # select the relevant slice of interaction matrix
        modmat = x_func(self.xmat, self.e_grid, *args)
        # Set lower triangular indices to 0. (should be not necessary)
        modmat[np.tril_indices(self.dim, -1)] = 0.

        return modmat

    def _set_mod_pprod(self, prim_pdg, sec_pdg, x_func, args):
        """Sets combination of projectile/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of
        ``prim_pdg`` is modified according to the function passed to
        :func:`InteractionYields.init_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
        """

        # Short cut for the pprod list
        mpli = self.mod_pprod
        pstup = (prim_pdg, sec_pdg)

        if config['use_isospin_sym'] and prim_pdg not in [2212, 2112]:
            raise Exception('Unsupported primary for isospin symmetries.')

        # if pstup not in mpli.keys():
        #     mpli[(pstup)] = {}

        if (x_func.__name__, args) in mpli[(pstup)]:
            info(5, ' no changes to particle production' +
                  ' modification matrix of {0}/{1} for {2},{3}'.format(
                      prim_pdg, sec_pdg, x_func.__name__, args))
            return False

        # Check function with same mode but different parameter is supplied
        for (xf_name, fargs) in mpli[pstup].keys():
            if (xf_name == x_func.__name__) and (fargs[0] == args[0]):
                info(1, 'Warning. If you modify only the value of a function,',
                        'unset and re-apply all changes')
                return False

        info(2, 'modifying modify particle production' +
                  ' matrix of {0}/{1} for {2},{3}').format(
                      prim_pdg, sec_pdg, x_func.__name__, args)

        kmat = self._gen_mod_matrix(x_func, *args)
        mpli[pstup][(x_func.__name__, args)] = kmat

        info(5, 'modification "strength"',
                  np.sum(kmat) / np.count_nonzero(kmat, dtype=np.float))

        if not config['use_isospin_sym']:
            return True

        prim_pdg, symm_pdg = 2212, 2112
        if prim_pdg == 2112:
            prim_pdg = 2112
            symm_pdg = 2212

        # p->pi+ = n-> pi-, p->pi- = n-> pi+
        if abs(sec_pdg) == 211:
            # Add the same mod to the isospin symmetric particle combination
            mpli[(symm_pdg, -sec_pdg)][('isospin', args)] = kmat

            # Assumption: Unflavored production coupled to the average
            # of pi+ and pi- production

            if np.any([p in self.projectiles for p in [221, 223, 333]]):

                unflv_arg = None
                if (prim_pdg, -sec_pdg) not in mpli:
                    # Only pi+ or pi- (not both) have been modified
                    unflv_arg = (args[0], 0.5 * args[1])

                if (prim_pdg, -sec_pdg) in mpli:
                    # Compute average of pi+ and pi- modification matrices
                    # Save the 'average' argument (just for meaningful printout)
                    for arg_name, arg_val in mpli[(prim_pdg, -sec_pdg)]:
                        if arg_name == args[0]:
                            unflv_arg = (args[0], 0.5 * (args[1] + arg_val))

                unflmat = self._gen_mod_matrix(x_func, *unflv_arg)

                # modify eta, omega, phi, 221, 223, 333
                for t in [(prim_pdg, 221), (prim_pdg, 223), (prim_pdg, 333),
                          (symm_pdg, 221), (symm_pdg, 223), (symm_pdg, 333)]:
                    mpli[t][('isospin', unflv_arg)] = unflmat

        # Charged and neutral kaons
        elif abs(sec_pdg) == 321:
            # approx.: p->K+ ~ n-> K+, p->K- ~ n-> K-
            mpli[(symm_pdg, sec_pdg)][('isospin', args)] = kmat
            k0_arg = (args[0], 0.5 * args[1])
            if (prim_pdg, -sec_pdg) in mpli:
                # Compute average of K+ and K- modification matrices
                # Save the 'average' argument (just for meaningful printout)
                for arg_name, arg_val in mpli[(prim_pdg, -sec_pdg)]:
                    if arg_name == args[0]:
                        k0_arg = (args[0], 0.5 * (args[1] + arg_val))

            k0mat = self._gen_mod_matrix(x_func, *k0_arg)

            # modify K0L/S
            for t in [(prim_pdg, 310), (prim_pdg, 130), (symm_pdg, 310),
                      (symm_pdg, 130)]:
                mpli[t][('isospin', k0_arg)] = k0mat

        elif abs(sec_pdg) == 411:
            ssec = np.sign(sec_pdg)
            mpli[(prim_pdg, ssec * 421)][('isospin', args)] = kmat
            mpli[(prim_pdg, ssec * 431)][('isospin', args)] = kmat
            mpli[(symm_pdg, sec_pdg)][('isospin', args)] = kmat
            mpli[(symm_pdg, ssec * 421)][('isospin', args)] = kmat
            mpli[(symm_pdg, ssec * 431)][('isospin', args)] = kmat

        # Leading particles
        elif abs(sec_pdg) == prim_pdg:
            mpli[(symm_pdg, symm_pdg)][('isospin', args)] = kmat
        elif abs(sec_pdg) == symm_pdg:
            mpli[(symm_pdg, prim_pdg)][('isospin', args)] = kmat
        else:
            raise Exception('No isospin relation found for secondary' +
                            str(sec_pdg))

        # Tell MCEqRun to regenerate the matrices if something has changed
        return True

    def print_mod_pprod(self):
        """Prints the active particle production modification.
        """

        for i, (prim_pdg, sec_pdg) in enumerate(sorted(self.mod_pprod)):
            for j, (argname, argv) in enumerate(self.mod_pprod[(prim_pdg,
                                                                sec_pdg)]):
                info(2, '{0}: {1} -> {2}, func: {3}, arg: {4}'.format(
                    i + j, prim_pdg, sec_pdg, argname, argv), no_caller=True)

    def get_xf_dist(self,
                    energy,
                    prim_pdg,
                    sec_pdg,
                    pos_only=True,
                    verbose=True,
                    **kwargs):
        """Returns :math:`dN/dx_{\rm F}` in c.m. for interaction energy close 
        to `energy` for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.
        
        Args:
            energy (float): approximate interaction energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm F}`, :math:`dN/dx_{\rm F}`
        """
        if not hasattr(self, '_ptav_sib23c'):
            # Load spline of average pt distribution as a funtion of log(E_lab) from sib23c
            import pickle
            from os.path import join
            self._ptav_sib23c = pickle.load(
                open(join(config['data_dir'], 'sibyll23c_aux.ppd'), 'rb'))[0]

        def xF(xL, Elab, ppdg):

            m = {2212: 0.938, 211: 0.139, 321: 0.493}
            mp = m[2212]

            Ecm = np.sqrt(2 * Elab * mp + 2 * mp**2)
            Esec = xL * Elab
            betacm = np.sqrt((Elab - mp) / (Elab + mp))
            gammacm = (Elab + mp) / Ecm
            avpt = self._ptav_sib23c[ppdg](
                np.log(np.sqrt(Elab**2) - m[np.abs(ppdg)]**2))

            xf = 2 * (-betacm * gammacm * Esec + gammacm *
                      np.sqrt(Esec**2 - m[np.abs(ppdg)]**2 - avpt**2)) / Ecm
            dxl_dxf = 1. / (2 * (
                -betacm * gammacm * Elab + xL * Elab**2 * gammacm / np.sqrt(
                    (xL * Elab)**2 - m[np.abs(ppdg)]**2 - avpt**2)) / Ecm)

            return xf, dxl_dxf

        eidx = (np.abs(self.e_grid - energy)).argmin()
        en = self.e_grid[eidx]
        info(2, 'Nearest energy, index: ', en, eidx, condition=verbose)
        m = self.get_y_matrix(prim_pdg, sec_pdg)
        xl_grid = self.e_grid[:eidx + 1] / en
        xl_dist = xl_grid * en * m[:eidx + 1, eidx] / np.diag(
            self.widths)[:eidx + 1]
        xf_grid, dxl_dxf = xF(xl_grid, en, sec_pdg)
        xf_dist = xl_dist * dxl_dxf

        if pos_only:
            xf_dist = xf_dist[xf_grid >= 0]
            xf_grid = xf_grid[xf_grid >= 0]
            return xf_grid, xf_dist

        return xf_grid, xf_dist

    def get_xlab_dist(self, energy, prim_pdg, sec_pdg, verbose=True, **kwargs):
        """Returns :math:`dN/dx_{\rm Lab}` for interaction energy close 
        to `energy` for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.
        
        Args:
            energy (float): approximate interaction energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self.e_grid - energy)).argmin()
        en = self.e_grid[eidx]
        info(2, 'Nearest energy, index: ', en, eidx, condition=verbose)

        m = self.get_y_matrix(prim_pdg, sec_pdg)
        xl_grid = self.e_grid[:eidx + 1] / en
        xl_dist = xl_grid * en * m[:eidx + 1, eidx] / np.diag(
            self.widths)[:eidx + 1]

        return xl_grid, xl_dist

    def set_interaction_model(self, interaction_model, force=False):
        """Selects an interaction model and prepares all internal variables.

        Args:
          interaction_model (str): interaction model name
          force (bool): forces reloading of data from file
        Raises:
          Exception: if invalid name specified in argument ``interaction_model``
        """

        interaction_model = normalize_hadronic_model_name(interaction_model)

        if not force and interaction_model == self.iam:
            info(5,'Model {0} already initialized.'.format(self.iam))
            return False
        else:
            self._load(interaction_model)

        if interaction_model != self.iam:
            raise Exception("InteractionYields(): No coupling matrices " +
                            "available for the selected interaction " +
                            "model: {0}.".format(interaction_model))
        return True

    def set_xf_band(self, xl_low_idx, xl_up_idx):
        """Limits interactions to certain range in :math:`x_{\\rm lab}`.

        Limit particle production to a range in :math:`x_{\\rm lab}` given
        by lower index, below which no particles are produced and an upper
        index, respectively. (Needs more clarification).

        Args:
          xl_low_idx (int): lower index of :math:`x_{\\rm lab}` value
          xl_up_idx (int): upper index of :math:`x_{\\rm lab}` value
        """

        if xl_low_idx >= 0 and xl_up_idx > 0:
            self.band = (xl_low_idx, xl_up_idx)
        else:
            self.band = None
            info(2, 'reset selection of x_lab band')
            return

        info(2, 'limiting Feynman x range to: {0:5.2e} - {1:5.2e}'.format(
                          (self.e_bins / self.e_bins[-1])[self.band[0]], 
                          (self.e_bins / self.e_bins[-1])[self.band[1]]))

    def is_yield(self, projectile, daughter):
        """Checks if a non-zero yield matrix exist for ``projectile``-
        ``daughter`` combination (deprecated)

        Args:
          projectile (int): PDG ID of projectile particle
          daughter (int): PDG ID of final state daughter/secondary particle
        Returns:
          bool: ``True`` if non-zero interaction matrix exists else ``False``
        """
        if projectile in self.projectiles and \
           daughter in self.secondary_dict[projectile]:
            return True
        else:
            info(2, 'no interaction matrix for {0}, {1}->{2}'.format(self.iam, projectile, daughter))
            return False

        return True

    def get_y_matrix(self, projectile, daughter):
        """Returns a ``DIM x DIM`` yield matrix.

        Args:
          projectile (int): PDG ID of projectile particle
          daughter (int): PDG ID of final state daughter/secondary particle
        Returns:
          numpy.array: yield matrix

        Note:
          In the current version, the matrices have to be multiplied by the
          bin widths. In later versions they will be stored with the multiplication
          carried out.
        """
        info(20, 'get_y_matrix entered for',projectile, daughter)

        if (config['adv_set']['disable_charm_pprod']
                and is_charm_pdgid):
            info(2, 'disabled particle production by', projectile)
            return self.no_interaction

        # The next line creates a copy, to prevent subsequent calls to modify
        # the original matrices stored in the dictionary.
        # @debug: probably performance bottleneck during init
        m = self.yields[(projectile, daughter)]  # .dot(self.widths)

        # For debugging purposes or plotting xlab distributions use this line instead
        # m = np.copy(self.yields[(projectile, daughter)])

        if config['adv_set']['disable_leading_mesons'] and abs(daughter) < 2000 \
                and (projectile, -daughter) in self.yields.keys():
            manti = self.yields[(projectile, -daughter)]  # .dot(self.widths)
            ie = 50
            info(2, 'sum in disable_leading_mesons', (np.sum(m[:, ie - 30:ie]) - np.sum(manti[:, ie - 30:ie])))

            if (np.sum(m[:, ie - 30:ie]) - np.sum(manti[:, ie - 30:ie])) > 0:
                info(2, 'inverting meson due to leading particle veto.', daughter, '->',
                          -daughter)
                m = manti
            else:
                info(2, 'no inversion since daughter not leading', daughter)
        else:
            info(20, 'no meson inversion in leading particle veto.', projectile, daughter)

        if (projectile, daughter) in self.mod_pprod.keys():
            info(2, 'using modified particle production for {0}/{1}'.format(projectile, daughter))
            m = np.copy(m)
            i = 0
            for args, mmat in self.mod_pprod[(projectile, daughter)].items():
                info(10, i, (projectile, daughter), args, np.sum(mmat), np.sum(m))
                i += 1
                m *= mmat

        if not self.band:
            return m
        else:
            # set all elements except those inside selected xf band to 0
            m = np.copy(m)
            m[np.tril_indices(self.dim, self.dim - self.band[1] - 1)] = 0.
            # if self.band[0] < 0:
            m[np.triu_indices(self.dim, self.dim - self.band[0])] = 0.
            return m

    def assign_yield_idx(self, projectile, projidx, daughter, dtridx, cmat):
        """Copies a subset, defined in tuples ``projidx`` and ``dtridx`` from
        the ``yield_matrix(projectile,daughter)`` into ``cmat``

        Args:
          projectile (int): PDG ID of projectile particle
          projidx (int,int): tuple containing index range relative
                             to the projectile's energy grid
          daughter (int): PDG ID of final state daughter/secondary particle
          dtridx (int,int): tuple containing index range relative
                            to the daughters's energy grid
          cmat (numpy.array): array reference to the interaction matrix
        """
        cmat[dtridx[0]:dtridx[1], projidx[0]:projidx[1]] = \
            self.get_y_matrix(projectile, daughter)[dtridx[0]:dtridx[1],
                                                    projidx[0]:projidx[1]]

    def _inject_custom_charm_model(self, model='MRS'):
        """Overwrites the charm production yields of the yield
        dictionary for the current interaction model with yields from
        a custom model.

        The function walks through all (projectile, charm_daughter)
        combinations and replaces the yield matrices with those from
        the ``model``.

        Args:
          model (str): charm model name

        Raises:
          NotImplementedError: if model string unknown.
        """

        from particletools.tables import SibyllParticleTable
        from MCEq.charm_models import MRS_charm, WHR_charm

        if model is None:
            return

        if self.charm_model and self.charm_model != model:
            # reload the yields from the main dictionary
            self.set_interaction_model(self.iam, force=True)

        sib = SibyllParticleTable()
        charm_modids = [
            sib.modid2pdg[modid] for modid in sib.mod_ids if abs(modid) >= 59
        ]
        del sib

        # Remove the charm interactions from the index
        new_index = {}
        for proj, secondaries in self.secondary_dict.iteritems():
            new_index[proj] = [
                idx for idx in secondaries if idx not in charm_modids
            ]

        self.secondary_dict = new_index

        if model == 'MRS':
            # Set charm production to zero
            cs = HadAirCrossSections(self.iam)
            mrs = MRS_charm(self.e_grid, cs)
            for proj in self.projectiles:
                for chid in charm_modids:
                    self.yields[(proj, chid)] = mrs.get_yield_matrix(
                        proj, chid).dot(self.widths)
                    # Update index
                    self.secondary_dict[proj].append(chid)

        elif model == 'WHR':

            cs_h_air = HadAirCrossSections('SIBYLL2.3')
            cs_h_p = HadAirCrossSections('SIBYLL2.3_pp')
            whr = WHR_charm(self.e_grid, cs_h_air)
            for proj in self.projectiles:
                cs_scale = np.diag(
                    cs_h_p.get_cs(proj) / cs_h_air.get_cs(proj)) * 14.5
                for chid in charm_modids:
                    self.yields[(proj, chid)] = whr.get_yield_matrix(
                        proj, chid).dot(self.widths) * 14.5
                    # Update index
                    self.secondary_dict[proj].append(chid)

        elif model == 'sibyll23_pl':
            cs_h_air = HadAirCrossSections('SIBYLL2.3')
            cs_h_p = HadAirCrossSections('SIBYLL2.3_pp')
            for proj in self.projectiles:
                cs_scale = np.diag(cs_h_p.get_cs(proj) / cs_h_air.get_cs(proj))
                for chid in charm_modids:
                    # rescale yields with sigma_pp/sigma_air to ensure
                    # that in a later step indeed sigma_{pp,ccbar} is taken

                    self.yields[(
                        proj, chid)] = self.yield_dict[self.iam + '_pl'][
                            (proj, chid)].dot(cs_scale).dot(self.widths) * 14.5
                    # Update index
                    self.secondary_dict[proj].append(chid)

        else:
            raise NotImplementedError(
                'InteractionYields:_inject_custom_charm_model()::' +
                ' Unsupported model')

        self.charm_model = model

        self._gen_particle_list()

    def __repr__(self):
        a_string = 'Possible (projectile,secondary) configurations:\n'
        for key in sorted(self.yields.keys()):
            if key not in ['evec', 'ebins']:
                a_string += str(key) + '\n'
        return a_string


class DecayYields(object):
    """Class for managing the dictionary of decay yield matrices.

    The class un-pickles a dictionary, which contains :math:`x`
    spectra of decay products/daughters, sampled from PYTHIA 8
    Monte Carlo.

    Args:
      mother_list (list, optional): list of particle mothers from
                                    interaction model
    """

    def __init__(self, mother_list=None, fname=None):
        #: (list) List of particles in the decay matrices
        self.particle_list = []

        self._load(mother_list, fname)

        self.particle_keys = self.mothers

    def _load(self, mother_list, fname):
        """Un-pickles the yields dictionary using the path specified as
        ``decay_fname`` in :mod:`mceq_config`.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join

        if fname:
            fname = join(config['data_dir'], fname)
        else:
            # Take the compact dictionary if "enabled" and no
            # file name forced
            if config['compact_mode']:
                fname = join(config['data_dir'],
                             'compact_' + config['decay_fname'])
            else:
                fname = join(config['data_dir'], config['decay_fname'])

        info(2, 'Loading file', fname)
        try:
            self.decay_dict = pickle.load(open(fname, 'rb'))
        except IOError:
            self._decompress(fname)
            self.decay_dict = pickle.load(open(fname, 'rb'))

        self.daughter_dict = self.decay_dict.pop('daughter_dict')
        self.widths = self.decay_dict.pop('widths')

        for mother in config["adv_set"]["disable_decays"]:
            info(2, 'switching off decays of {0}.'.format(mother))
            self.daughter_dict.pop(mother)

        # Restrict accessible decay yields to mother particles from mother_list
        if mother_list is None:
            self.mothers = self.daughter_dict.keys()
        else:
            for m in mother_list:
                if (m not in self.daughter_dict.keys()
                        and abs(m) not in [2212, 11, 12, 14, 22, 7012, 7014]):
                    info(1, 'Warning: no decay distributions for {0} found.'.format(m))

            # Remove unused particle species from index in compact mode
            if config["compact_mode"]:
                for p in self.daughter_dict.keys():
                    if abs(p) not in mother_list and abs(p) not in [
                            7113, 7213, 7313
                    ]:
                        _ = self.daughter_dict.pop(p)

            self.mothers = self.daughter_dict.keys()

        self._gen_particle_list(mother_list)

    def _gen_particle_list(self, mother_list):
        """Saves a list of all particle species in the decay dictionary.

        """

        # Look up all particle species that a supported by selected model

        for p, l in self.daughter_dict.iteritems():
            self.particle_list += [p]
            self.particle_list += l
        self.particle_list = list(set(self.particle_list))

    def _gen_index(self, decay_dict):
        """Generates index of mother-daughter relationships.

        This function is called once after un-pickling. In future
        versions this index will be part of the pickled dictionary.
        """
        temp = np.unique(zip(*decay_dict.keys())[0])
        # Filter out the non numerical strings from this list
        mothers = []
        for mo in temp:
            try:
                mothers.append(int(mo))
            except ValueError:
                continue

        e_bins = decay_dict['ebins']
        widths = np.diag(e_bins[1:] - e_bins[:-1])
        e_grid = np.sqrt(e_bins[1:] * e_bins[:-1])

        # This will be the dictionary for the index
        daughter_dict = {}

        # New dictionary to replace decay_dict
        new_dict = {}

        for mother in mothers:
            daughter_dict[mother] = []

        for key, mat in decay_dict.iteritems():
            try:
                mother, daughter = key
            except ValueError:
                info(5,'Skip additional info', key)
                # Copy additional items to the new dictionary
                new_dict[key] = mat
                continue

            if np.sum(mat) > 0:
                if daughter not in daughter_dict[mother]:
                    daughter_dict[mother].append(daughter)
                    # Multiply by widths (energy bin widths with matrices)
                    new_dict[key] = (mat.T).dot(widths)

        # special treatment for muons, which should decay even if they
        # have an alias ID
        # the ID 7313 not included, since it's "a copy of"
        for alias in [7013, 7113, 7213, 7313]:
            # if 13 not in config["adv_set"]["disable_decays"]:
            daughter_dict[alias] = daughter_dict[13]
            for d in daughter_dict[alias]:
                new_dict[(alias, d)] = new_dict[(13, d)]
            # if -13 not in config["adv_set"]["disable_decays"]:
            daughter_dict[-alias] = daughter_dict[-13]
            for d in daughter_dict[-alias]:
                new_dict[(-alias, d)] = new_dict[(-13, d)]

        new_dict['mothers'] = mothers
        new_dict['widths'] = widths
        new_dict['ebins'] = e_bins
        new_dict['evec'] = e_grid
        new_dict['daughter_dict'] = daughter_dict

        return new_dict

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        The method calls :func:`DecayYields._gen_index` to browse
        through the file, to create an index of mother daughter relations
        and to carry out some pre-computations. In the end an uncompressed
        file is stored including the index as a dictionary.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(
                self.__class__.__name__ +
                '::_decompress():: File {0} not found.'.format(fcompr))

        info(2, 'Decompressing', fcompr)

        # Generate index of mother daughter relations and
        # multiply with bin widths
        new_dict = self._gen_index(pickle.load(bz2.BZ2File(fcompr)))

        # Dump the file in uncompressed form
        info(2, 'Saving to', fname)

        pickle.dump(new_dict, open(fname, 'wb'), protocol=-1)

    def get_d_matrix(self, mother, daughter):
        """Returns a ``DIM x DIM`` decay matrix.

        Args:
          mother (int): PDG ID of mother particle
          daughter (int): PDG ID of final state daughter particle
        Returns:
          numpy.array: decay matrix

        Note:
          In the current version, the matrices have to be multiplied by the
          bin widths. In later versions they will be stored with the multiplication
          carried out.
        """
        info(5, 'trying to get empty matrix {0} -> {1}'.format(mother, daughter), 
            condition=not self.is_daughter(mother, daughter))

        return self.decay_dict[(mother, daughter)]

    def assign_d_idx(self, mother, moidx, daughter, dtridx, dmat):
        """Copies a subset, defined in tuples ``moidx`` and ``dtridx`` from
        the ``decay_matrix(mother,daughter)`` into ``dmat``

        Args:
          mother (int): PDG ID of mother particle
          moidx (int,int): tuple containing index range relative
                             to the mothers's energy grid
          daughter (int): PDG ID of final state daughter/secondary particle
          dtridx (int,int): tuple containing index range relative
                            to the daughters's energy grid
          dmat (numpy.array): array reference to the decay matrix
        """

        dmat[dtridx[0]:dtridx[1], moidx[0]:moidx[1]] = \
            self.get_d_matrix(mother, daughter)[dtridx[0]:dtridx[1],
                                                moidx[0]:moidx[1]]

    def is_daughter(self, mother, daughter):
        """Checks if ``daughter`` is a decay daughter of ``mother``.

        Args:
          mother (int): PDG ID of projectile particle
          daughter (int): PDG ID of daughter particle
        Returns:
          bool: ``True`` if ``daughter`` is daughter of ``mother``
        """
        if (mother not in self.daughter_dict.keys()
                or daughter not in self.daughter_dict[mother]):
            return False
        else:
            return True

    def daughters(self, mother):
        """Checks if ``mother`` decays and returns the list of daughter particles.

        Args:
          mother (int): PDG ID of projectile particle
        Returns:
          list: PDG IDs of daughter particles
        """
        if mother not in self.daughter_dict.keys():
            info(5, 'requesting daughter list for stable or not existing mother:', mother)
            return []
        return self.daughter_dict[mother]

    def __repr__(self):
        a_string = 'Possible (mother,daughter) configurations:\n'
        for key in sorted(self.decay_dict.keys()):
            a_string += str(key) + '\n'
        return a_string


class HadAirCrossSections(object):
    """Class for managing the dictionary of hadron-air cross-sections.

    The class unpickles a dictionary, which contains proton-air,
    pion-air and kaon-air cross-sections tabulated on the common
    energy grid.

    Args:
      interaction_model (str): name of the interaction model
    """
    #: unit - :math:`\text{GeV} \cdot \text{fm}`
    GeVfm = 0.19732696312541853
    #: unit - :math:`\text{GeV} \cdot \text{cm}`
    GeVcm = GeVfm * 1e-13
    #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
    GeV2mbarn = 10.0 * GeVfm**2
    #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
    mbarn2cm2 = GeVcm**2 / GeV2mbarn

    def __init__(self, interaction_model):
        #: current interaction model name
        self.iam = None
        #: current energy grid
        self.egrid = None

        self._load()

        interaction_model = normalize_hadronic_model_name(interaction_model)
        if interaction_model != None:
            self.set_interaction_model(interaction_model)
        else:
            # Set some default interaction model to allow for cross-sections
            self.set_interaction_model('SIBYLL2.3')

    def _load(self):
        """Un-pickles a dictionary using the path specified as
        ``decay_fname`` in :mod:`mceq_config`.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join
        fname = join(config['data_dir'], config['cs_fname'])
        try:
            self.cs_dict = pickle.load(open(fname, 'rb'))
        except IOError:
            self._decompress(fname)
            self.cs_dict = pickle.load(open(fname, 'rb'))

        # normalise hadronic model names
        old_keys = [k for k in self.cs_dict if k != "EVEC"]
        for old_key in old_keys:
            new_key = normalize_hadronic_model_name(old_key)
            self.cs_dict[new_key] = self.cs_dict.pop(old_key)

        self.egrid = self.cs_dict['EVEC']

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(
                self.__class__.__name__ +
                '::_decompress():: File {0} not found.'.format(fcompr))

        info(2, 'Decompressing', fcompr)

        new_dict = pickle.load(bz2.BZ2File(fcompr))

        # Dump the file in uncompressed form
        info(2, 'Saving to', fname)
        pickle.dump(new_dict, open(fname, 'wb'), protocol=-1)

    def set_interaction_model(self, interaction_model):
        """Selects an interaction model and prepares all internal variables.

        Args:
          interaction_model (str): interaction model name
        Raises: 
          Exception: if invalid name specified in argument ``interaction_model``
        """
        # Remove the _compact suffix, since this does not affect cross sections
        info(5, 'Using cross sections of original model in compact mode')
        interaction_model = normalize_hadronic_model_name(interaction_model)
        interaction_model = interaction_model.split('_compact')[0]

        if interaction_model == self.iam:
            info(5, 'Model {0} already initialized.'.format(self.iam))
            return
        if interaction_model in self.cs_dict.keys():
            self.iam = interaction_model

        else:
            info(-1, 'Available interaction models:', self.cs_dict.keys())
            raise Exception('No cross-sections for ' +
                'interaction model {0} available.'.format(interaction_model))

        self.cs = self.cs_dict[self.iam]

    def __getitem__(self, projectile):
        """Return the cross section in :math:`\\text{cm}^2` as a dictionary lookup."""
        return self.get_cs(projectile)

    def get_cs(self, projectile, mbarn=False):
        """Returns inelastic ``projectile``-air cross-section
        :math:`\\sigma_{inel}^{proj-Air}(E)` as vector spanned over
        the energy grid.

        Args:
          projectile (int): PDG ID of projectile particle
          mbarn (bool,optional): if ``True``, the units of the cross-section
                                 will be :math:`mbarn`, else :math:`\\text{cm}^2`

        Returns:
          numpy.array: cross-section in :math:`mbarn` or :math:`\\text{cm}^2`
        """

        message_templ = 'HadAirCrossSections(): replacing {0} with {1} cross-section'
        scale = 1.0
        if not mbarn:
            scale = self.mbarn2cm2
        if abs(projectile) in self.cs.keys():
            return scale * self.cs[projectile]
        elif abs(projectile) in [411, 421, 431, 15]:
            info(7, message_templ.format('D', 'K+-'))
            return scale * self.cs[321]
        elif abs(projectile) in [4332, 4232, 4132]:
            info(7, message_templ.format('charmed baryon', 'nucleon'))
            return scale * self.cs[2212]
        elif abs(projectile) > 2000 and abs(projectile) < 5000:
            info(7, message_templ.format(projectile, 'nucleon'))
            return scale * self.cs[2212]
        elif 5 < abs(projectile) < 23 or 7000 < abs(projectile) < 7500:
            info(7, 'returning 0 cross-section for lepton', projectile)
            return np.zeros_like(self.cs[2212])
        else:
            info(7, message_templ.format(projectile, 'pion'))
            return scale * self.cs[211]

    def __repr__(self):
        a_string = 'HadAirCrossSections() available for the projectiles: \n'
        for key in sorted(self.cs.keys()):
            a_string += str(key) + '\n'
        return a_string
