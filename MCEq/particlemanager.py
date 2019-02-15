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
from MCEq.misc import info

from particletools.tables import PYTHIAParticleData
info(5, 'Initialization of PYTHIAParticleData object')
_pdata = PYTHIAParticleData()

class MCEqParticle(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): handle to an instance of :class:`ParticleDataTool.SibyllParticleTable`
      pythia_db (object): handle to an instance of :class:`ParticleDataTool.PYTHIAParticleData`
      cs_db (object): handle to an instance of :class:`InteractionYields`
      d (int): dimension of the energy grid
    """

    def __init__(self, pdgid, particle_db, cs_db, d):

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
        self.sname = None
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
        self.pythia_db = _pdata

        # TODO: move this check to internal variable self.is_stable, or so
        # if pdgid in config["adv_set"]["disable_decays"]:
        #     self.pythia_db.force_stable(self.pdgid)

        self.cs = cs_db
        self.d = d

        self.E_crit = self.critical_energy()
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

    def hadridx(self):
        """Returns index range where particle behaves as hadron.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (self.mix_idx, self.d)

    def residx(self):
        """Returns index range where particle behaves as resonance.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (0, self.mix_idx)

    def lidx(self):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`MCEqRun.phi`
        """
        return self.mceqidx * self.d

    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.mceqidx + 1) * self.d

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
            dlen = self.pythia_db.mass(self.pdgid) / \
                self.pythia_db.ctau(self.pdgid) / E
            if cut:
                dlen[0:self.mix_idx] = 0.
            return 0.9966 * dlen  # Correction for bin average
        except ZeroDivisionError:
            return np.ones(self.d) * np.inf

    def inverse_interaction_length(self, cs=None):
        """Returns inverse interaction length for A_target given by config.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_target = config['A_target'] * 1.672621 * 1e-24  # <A> * m_proton [g]
        return np.ones(self.d) * self.cs.get_cs(self.pdgid) / m_target

    def critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.

        Returns:
          (float): :math:`\\frac{m\\ 6.4 \\text{km}}{c\\tau}` in GeV
        """
        try:
            return self.pythia_db.mass(self.pdgid) * 6.4e5 / \
                self.pythia_db.ctau(self.pdgid)
        except ZeroDivisionError:
            return np.inf

    def calculate_mixing_energy(self, e_grid, no_mix=False):
        """Calculates interaction/decay length in Air and decides if
        the particle has resonance and/or hadron behavior.

        Class attributes :attr:`is_mixed`, :attr:`E_mix`, :attr:`mix_idx`,
        :attr:`is_resonance` are calculated here.

        Args:
          e_grid (numpy.array): energy grid of size :attr:`d`
          no_mix (bool): if True, mixing is disabled and all particles
                         have either hadron or resonances behavior.
          max_density (float): maximum density on the integration path (largest
                               decay length)
        """

        cross_over = config["hybrid_crossover"]
        max_density = config['max_density']
        if abs(self.pdgid) in [2212]:
            self.mix_idx = 0
            self.is_mixed = False
            return
        d = self.d

        inv_intlen = self.inverse_interaction_length()
        inv_declen = self.inverse_decay_length(e_grid)

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
            d_tilde = 1 / self.inverse_decay_length(e_grid)

            # multiply with maximal density encountered along the
            # integration path
            ldec = d_tilde * max_density
            threshold = ldec / lint
        else:
            threshold = np.inf
            no_mix = True

        if np.max(threshold) < cross_over:
            self.mix_idx = d - 1
            self.E_mix = e_grid[self.mix_idx]
            self.is_mixed = False
            self.is_resonance = True

        elif np.min(threshold) > cross_over or no_mix:
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
        else:
            self.mix_idx = np.where(ldec / lint > cross_over)[0][0]
            self.E_mix = e_grid[self.mix_idx]
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

    def __init__(self, pdgid_list, egrid_dim):
        # (dict) Dimension of primary grid
        self.grid_dims = {'default': egrid_dim}
        # Particle index shortcuts
        #: (dict) Converts Neucosma ID to index in state vector
        self.pdgid2mceqidx = {}
        #: (dict) Converts particle name to index in state vector
        self.sname2mceqidx = {}
        #: (dict) Converts Neucosma ID to reference of
        # :class:`particlemanager.MCEqParticle`
        self.pdgid2sref = {}
        #: (dict) Converts particle name to reference of
        #:class:`particlemanager.MCEqParticle`
        self.sname2sref = {}
        #: (dict) Converts prince index to reference of
        #:class:`particlemanager.MCEqParticle`
        self.mceqidx2sref = {}
        #: (dict) Converts index in state vector to Neucosma ID
        self.mceqidx2pdgid = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`particlemanager.MCEqParticle`
        self.mceqidx2pname = {}
        #: (int) Total number of species
        self.nspec = 0

        self._init_particle_tables(pdgid_list)
        self._init_mapping_tables()

    def _init_mapping_tables(self):
        for s in self.species_refs:
            self.pdgid2mceqidx[s.pdgid] = s.mceqidx
            self.sname2mceqidx[s.sname] = s.mceqidx
            self.mceqidx2pdgid[s.mceqidx] = s.pdgid
            self.mceqidx2pname[s.mceqidx] = s.sname
            self.pdgid2sref[s.pdgid] = s
            self.mceqidx2sref[s.mceqidx] = s
            self.sname2sref[s.sname] = s

        self.nspec = len(self.species_refs)

    def _init_particle_tables(self, particle_list=None):

        self.particle_species, self.cascade_particles, self.resonances = \
            self._gen_list_of_particles(custom_list=particle_list)

        # Further short-cuts depending on previous initializations
        self.n_tot_species = len(self.cascade_particles)

        self.dim_states = self.d * self.n_tot_species

        self.muon_selector = np.zeros(self.dim_states, dtype='bool')
        for p in self.particle_species:
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
                self.muon_selector[p.lidx():p.uidx()] = True

        self.e_weight = np.array(
            self.n_tot_species * list(self.y.e_bins[1:] - self.y.e_bins[:-1]))

        self.solution = np.zeros(self.dim_states)

        # Initialize empty state (particle density) vector
        self.phi0 = np.zeros(self.dim_states).astype(self.fl_pr)

        self._init_alias_tables()
        self._init_muon_energy_loss()

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

        particles = None

        if custom_list:
            try:
                # Assume that particle list contains particle names
                particles = [
                    self.modtab.modname2pdg[pname] for pname in custom_list
                ]
            except KeyError:
                # assume pdg indices
                particles = custom_list
            except:
                raise Exception("custom particle list not understood:" +
                                ','.join(custom_list))

            particles += self.modtab.leptons
        else:
            particles = self.modtab.baryons + self.modtab.mesons + self.modtab.leptons

        # Remove duplicates
        particles = list(set(particles))

        particle_list = [
            MCEqParticle(h, self.modtab, self.pd, self.cs, self.d)
            for h in particles
        ]

        particle_list.sort(key=lambda x: x.E_crit, reverse=False)

        for p in particle_list:
            p.calculate_mixing_energy(self._e_grid, self.adv_set['no_mixing'])

        cascade_particles = [p for p in particle_list if not p.is_resonance]
        resonances = [p for p in particle_list if p.is_resonance]

        for mceqidx, h in enumerate(cascade_particles):
            h.mceqidx = mceqidx

        return cascade_particles + resonances, cascade_particles, resonances

    # def _gen_species(self, pdgid_list):
    #     info(4, "Generating list of species.")

    #     # pdgid_list += spec_data["non_nuclear_species"]

    #     # Make sure list is unique and sorted
    #     pdgid_list = sorted(list(set(pdgid_list)))

    #     self.species_refs = []
    #     # Define position in state vector (mceqidx) by simply
    #     # incrementing it with the (sorted) list of Neucosma IDs
    #     for mceqidx, pdgid in enumerate(pdgid_list):
    #         info(
    #             4, "Appending species {0} at position {1}".format(
    #                 pdgid, mceqidx))
    #         self.species_refs.append(
    #             MCEqParticle(pdgid, mceqidx, self.grid_dims['default']))

    #     self.known_species = [s.pdgid for s in self.species_refs]
    #     self.redist_species = [
    #         s.pdgid for s in self.species_refs if s.has_redist
    #     ]
    #     self.boost_conserv_species = [
    #         s.pdgid for s in self.species_refs if not s.has_redist
    #     ]

        def add_grid(self, grid_tag, dimension):
        """Defines additional grid dimensions under a certain tag.

        Propagates changes to this variable to all known species.
        """
        info(2, 'New grid_tag', grid_tag, 'with dimension', dimension)
        self.grid_dims[grid_tag] = dimension

        for s in self.species_refs:
            s.grid_dims = self.grid_dims

    def __repr__(self):
        str_out = ""
        ident = 3 * ' '
        for s in self.species_refs:
            str_out += s.sname + '\n' + ident
            str_out += 'NCO id : ' + str(s.pdgid) + '\n' + ident
            str_out += 'PriNCe idx : ' + str(s.mceqidx) + '\n\n'

        return str_out