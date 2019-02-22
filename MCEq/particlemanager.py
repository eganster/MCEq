# -*- coding: utf-8 -*-
"""
:mod:`MCEq.particlemanager` --- management of particle objects
==============================================================

This module includes code for bookkeeping, interfacing and
validating data structures:

- :class:`MCEqParticle` bundles different particle properties for simpler
  usage in :class:`MCEqRun`


"""
import numpy as np
from mceq_config import config
from MCEq.misc import info, print_in_rows

from particletools.tables import PYTHIAParticleData
info(5, 'Initialization of PYTHIAParticleData object')
_pdata = PYTHIAParticleData()

class MCEqParticle(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): handle to an instance of :class:`ParticleDataTool.SibyllParticleTable`
      egrid (np.array, optional): energy grid (centers)
      cs_db (object, optional): reference to an instance of :class:`InteractionYields`
    """

    def __init__(self, pdgid, particle_db, e_grid=None, cs_db=None):

        #: (bool) particle is a hadron (meson or baryon)
        self.is_hadron = False
        #: (bool) particle is a meson
        self.is_meson = False
        #: (bool) particle is a baryon
        self.is_baryon = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (bool) if it's an electromagnetic particle
        self.is_em = False
        #: (bool) particle is a lepton
        self.is_charged = False
        #: (bool) particle is a nucleus
        self.is_nucleus = False
        #: (bool) particle is stable
        self.is_stable = True
        #: (float) lifetime
        self.lifetime = np.inf
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.is_alias = False
        #: (str) species name in string representation
        self.name = None
        #: decay channels if any
        self.decay_channels = {}
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = 1, None, None
        #: Mass in atomic units or GeV
        self.mass = None
        #: (bool) particle has both, hadron and resonance properties
        self.is_mixed = False
        #: (bool) if particle has just resonance behavior
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdgid = pdgid
        #: (int) MCEq ID
        self.mceqidx = -1

        #: (float) mixing energy, transition between hadron and resonance behavior
        self.E_mix = 0
        #: (int) energy grid index, where transition between hadron and resonance occurs
        self.mix_idx = 0
        #: (float) critical energy in air at the surface
        self.E_crit = 0

        self.particle_db = particle_db

        # # TODO: move this check to internal variable self.is_stable, or so
        # # if pdgid in config["adv_set"]["disable_decays"]:
        # #     _pdata.force_stable(self.pdgid)

        self.name = particle_db.pdg2modname[pdgid]

        if pdgid in particle_db.mesons:
            self.is_hadron = True
            self.is_meson = True
        elif pdgid in particle_db.baryons:
            self.is_hadron = True
            self.is_baryon = True
        else:
            self.is_lepton = True
            if abs(pdgid) > 22:
                self.is_alias = True

        # Energy grid dependent inits
        if e_grid is not None and cs_db is not None:
            self.cs = cs_db
            self._e_grid = e_grid
            self._d = len(e_grid)

            self._critical_energy()
            self._calculate_mixing_energy()

    @property
    def hadridx(self):
        """Returns index range where particle behaves as hadron.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (self.mix_idx, self._d)

    @property
    def residx(self):
        """Returns index range where particle behaves as resonance.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (0, self.mix_idx)

    @property
    def lidx(self):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`MCEqRun.phi`
        """
        return self.mceqidx * self._d

    @property
    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.mceqidx + 1) * self._d

    def inverse_decay_length(self, E, cut=True):
        """Returns inverse decay length (or infinity (np.inf), if
        particle is stable), where the air density :math:`\\rho` is
        factorized out.

        Args:
          E (float) : energy in laboratory system in GeV
          cut (bool): set to zero in 'resonance' regime
        Returns:
          (float): :math:`\\frac{\\rho}{\\lambda_{dec}}` in 1/cm
        """
        try:
            dlen = _pdata.mass(self.pdgid) / \
                _pdata.ctau(self.pdgid) / E
            if cut:
                dlen[0:self.mix_idx] = 0.
            return 0.9966 * dlen  # Correction for bin average
        except ZeroDivisionError:
            return np.ones(self._d) * np.inf

    def inverse_interaction_length(self, cs=None):
        """Returns inverse interaction length for A_target given by config.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_target = config['A_target'] * 1.672621 * 1e-24  # <A> * m_proton [g]
        return np.ones(self._d) * self.cs.get_cs(self.pdgid) / m_target
    
    def _critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.

        Returns:
          (float): :math:`\\frac{m\\ 6.4 \\text{km}}{c\\tau}` in GeV
        """
        try:
            self.E_crit = _pdata.mass(self.pdgid) * 6.4e5 / _pdata.ctau(self.pdgid)
        except ZeroDivisionError:
            self.E_crit = np.inf

    def _calculate_mixing_energy(self):
        """Calculates interaction/decay length in Air and decides if
        the particle has resonance and/or hadron behavior.

        Class attributes :attr:`is_mixed`, :attr:`E_mix`, :attr:`mix_idx`,
        :attr:`is_resonance` are calculated here. If the option `no_mixing`
        is set in config['adv_config'] particle is forced to be a resonance
        or hadron behavior.

        Args:
          e_grid (numpy.array): energy grid of size :attr:`d`
          max_density (float): maximum density on the integration path (largest
                               decay length)
        """

        cross_over = config["hybrid_crossover"]
        max_density = config['max_density']
        no_mix = config["adv_set"]['no_mixing']

        if abs(self.pdgid) in [2212]:
            self.mix_idx = 0
            self.is_mixed = False
            return
        d = self._d

        inv_intlen = self.inverse_interaction_length()
        inv_declen = self.inverse_decay_length(self._e_grid)

        if (not np.any(inv_declen > 0.) or abs(
                self.pdgid) in config["adv_set"]["exclude_from_mixing"]):
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
            return

        if np.abs(self.pdgid) in config["adv_set"]["force_resonance"]:
            threshold = 0.
        elif np.any(inv_intlen > 0.):
            lint = np.ones(d) / inv_intlen
            d_tilde = 1 / self.inverse_decay_length(self._e_grid)

            # multiply with maximal density encountered along the
            # integration path
            ldec = d_tilde * max_density
            threshold = ldec / lint
        else:
            threshold = np.inf
            no_mix = True

        if np.max(threshold) < cross_over:
            self.mix_idx = d - 1
            self.E_mix = self._e_grid[self.mix_idx]
            self.is_mixed = False
            self.is_resonance = True

        elif np.min(threshold) > cross_over or no_mix:
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
        else:
            self.mix_idx = np.where(ldec / lint > cross_over)[0][0]
            self.E_mix = self._e_grid[self.mix_idx]
            self.is_mixed = True
            self.is_resonance = False

    def __repr__(self):
        a_string = ("""
        {0}:
        is_hadron   : {1}
        is_mixed    : {2}
        is_resonance: {3}
        is_lepton   : {4}
        is_alias    : {5}
        E_mix       : {6:2.1e}\n""").format(
            self.name, self.is_hadron, self.is_mixed, self.is_resonance,
            self.is_lepton, self.is_alias, self.E_mix)
        return a_string


class ParticleManager(object):
    """Provides a database with particle and species.
    
    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """

    def __init__(self, pdgid_list, e_grid, cs_db, mod_table=None):
        # (dict) Dimension of primary grid
        self._e_grid = e_grid
        self._d = len(e_grid)
        # Particle index shortcuts
        #: (dict) Converts Neucosma ID to index in state vector
        self.pdg2mceqidx = {}
        #: (dict) Converts particle name to index in state vector
        self.pname2mceqidx = {}
        #: (dict) Converts Neucosma ID to reference of
        # :class:`particlemanager.MCEqParticle`
        self.pdg2pref = {}
        #: (dict) Converts particle name to reference of
        #:class:`particlemanager.MCEqParticle`
        self.pname2pref = {}
        #: (dict) Converts prince index to reference of
        #:class:`particlemanager.MCEqParticle`
        self.mceqidx2pref = {}
        #: (dict) Converts index in state vector to Neucosma ID
        self.mceqidx2pdg = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`particlemanager.MCEqParticle`
        self.mceqidx2pname = {}
        #: (int) Total number of species
        self.nspec = 0

        if mod_table is None:
            from particletools.tables import SibyllParticleTable
            self.mod_table = SibyllParticleTable()
        else:
            self.mod_table = mod_table

        # Cross section database
        self._cs_db = cs_db

        self._init_particle_tables(pdgid_list)

    def _init_particle_tables(self, particle_list=None):

        
        self._init_categories(particle_pdg_list=particle_list)

        # Further short-cuts depending on previous initializations
        self.n_species = len(self.cascade_particles)
        self.dim_states = self._d * self.n_species

        self.muon_selector = np.zeros(self.dim_states, dtype='bool')
        for p in self.all_particles:
            try:
                mceqidx = p.mceqidx
            except AttributeError:
                mceqidx = -1
            self.pdg2mceqidx[p.pdgid] = mceqidx
            self.pname2mceqidx[p.name] = mceqidx
            self.mceqidx2pdg[mceqidx] = p.pdgid
            self.mceqidx2pname[mceqidx] = p.name
            self.pdg2pref[p.pdgid] = p
            self.pname2pref[p.name] = p

            # Select all positions of muon species in the state vector
            if abs(p.pdgid) % 1000 % 100 % 13 == 0 and not (100 < abs(p.pdgid)
                                                            < 7000):
                self.muon_selector[p.lidx:p.uidx] = True

        self._init_alias_tables()
        # self._init_muon_energy_loss()

        self.print_particle_tables(2)

    def _init_categories(self, particle_pdg_list):
        """Determines the list of particles for calculation and
        returns lists of instances of :class:`data.MCEqParticle` .

        The particles which enter this list are those, which have a
        defined index in the SIBYLL 2.3 interaction model. Included are
        most relevant baryons and mesons and some of their high mass states.
        More details about the particles which enter the calculation can
        be found in :mod:`ParticleDataTool`.

        Returns:
          (tuple of lists of :class:`data.MCEqParticle`): (all particles,
          cascade particles, resonances)
        """
        from MCEq.particlemanager import MCEqParticle

        info(5, "Generating particle list.")

        if particle_pdg_list is not None:
            particles = particle_pdg_list
            particles += self.mod_table.leptons
        else:
            particles = self.mod_table.baryons + self.mod_table.mesons + self.mod_table.leptons

        # Remove duplicates
        particles = list(set(particles))

        # Initialize particle objects
        particle_list = [
            MCEqParticle(h, self.mod_table, self._e_grid, self._cs_db)
            for h in particles
        ]

        # Sort by critical energy (= int_len ~== dec_length ~ int_cs/tau)
        particle_list.sort(key=lambda x: x.E_crit, reverse=False)

        # Cascade particles will "live" on the grid and have an mceqidx assigned
        cascade_particles = [p for p in particle_list if not p.is_resonance]

        # These particles will only exist implicitely and integated out
        resonances = [p for p in particle_list if p.is_resonance]

        # Assign an mceqidx (position in state vector) to each explicit particle
        for mceqidx, h in enumerate(cascade_particles):
            h.mceqidx = mceqidx

        self.cascade_particles = cascade_particles
        self.resonances = resonances
        self.all_particles = cascade_particles + resonances
    
    def add_tracking_particle(self, parent_list, child, alias):
        """Allows tracking decay and particle production chains.

        Replaces previous ``obs_particle`` function that allowed to track
        only leptons from decays certain particles. This present feature
        removes the special PDG IDs 71XX, 72XX, etc and allows to define
        any channel like::
        
            $ particleManagerInstance.add_tracking_particle([211], 14, 'pi_numu')
        
        This will store muon neutrinos from pion decays under the alias 'pi_numu'.
        Multiple parents are allowed, as well::

            $ particleManagerInstance.add_tracking_particle(
                [411, 421, 431], 14, 'D_numu')

        Args:

            alias (str): Name alias under which the result is accessible in get_solution
            parents (list): list of parent particle PDG ID's
            child (int): Child particle
        """

        pass

    def _init_alias_tables(self):
        r"""Sets up the functionality of aliases and defines the meaning of
        'prompt'.

        The identification of the last mother particle of a lepton is implemented
        via index aliases. I.e. the PDG index of muon neutrino 14 is transformed
        into 7114 if it originates from decays of a pion, 7214 in case of kaon or
        7014 if the mother particle is very short lived (prompt). The 'very short lived'
        means that the critical energy :math:`\varepsilon \ge \varepsilon(D^\pm)`.
        This includes all charmed hadrons, as well as resonances such as :math:`\eta`.

        The aliases for the special ``obs_`` category are also initialized here.
        """
        info(5, "Initializing links to alias IDs.")

        self.alias_table = {}
        prompt_ids = []
        for p in self.all_particles:
            if p.is_lepton or p.is_alias or p.pdgid < 0:
                continue
            if 411 in self.pdg2pref and p.E_crit >= self.pdg2pref[411].E_crit:
                prompt_ids.append(p.pdgid)
        for lep_id in [12, 13, 14, 16]:
            self.alias_table[(211, lep_id)] = 7100 + lep_id  # pions
            self.alias_table[(321, lep_id)] = 7200 + lep_id  # kaons
            for pr_id in prompt_ids:
                self.alias_table[(pr_id, lep_id)] = 7000 + lep_id  # prompt

        # # check if leptons coming from mesons located in obs_ids should be
        # # in addition scored in a separate category (73xx)
        # self.obs_table = {}
        # if self.obs_ids is not None:
        #     for obs_id in self.obs_ids:
        #         if obs_id in self.pdg2pref.keys():
        #             self.obs_table.update({
        #                 (obs_id, 12): 7312,
        #                 (obs_id, 13): 7313,
        #                 (obs_id, 14): 7314,
        #                 (obs_id, 16): 7316
        #             })

    def __repr__(self):
        str_out = ""
        ident = 3 * ' '
        for s in self.all_particles:
            str_out += s.name + '\n' + ident
            str_out += 'PDG id : ' + str(s.pdgid) + '\n' + ident
            str_out += 'MCEq idx : ' + str(s.mceqidx) + '\n\n'

        return str_out

    def print_particle_tables(self, min_dbg_lev=2):

        info(min_dbg_lev, "\nHadrons and stable particles:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name for p in self.all_particles
            if p.is_hadron and not p.is_resonance and not p.is_mixed
        ])

        info(min_dbg_lev, "\nMixed:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [p.name for p in self.all_particles if p.is_mixed])

        info(min_dbg_lev, "\nResonances:\n", no_caller=True)
        print_in_rows(min_dbg_lev, 
            [p.name for p in self.all_particles if p.is_resonance])

        info(min_dbg_lev, "\nLeptons:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name for p in self.all_particles
            if p.is_lepton and not p.is_alias
        ])
        info(min_dbg_lev, "\nAliases:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [p.name for p in self.all_particles if p.is_alias])

        info(
            min_dbg_lev,
            "\nTotal number of species:",
            self.n_species,
            no_caller=True)

        # list particle indices
        if False:
            info(
                10,
                "Particle matrix indices:",
                no_caller=True)
            some_index = 0
            for p in self.cascade_particles:
                for i in xrange(self._d):
                    info(
                        10,
                        p.name + '_' + str(i),
                        some_index,
                        no_caller=True)
                    some_index += 1