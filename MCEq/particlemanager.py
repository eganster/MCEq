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

    def __init__(self, pdgid, particle_db, pythia_db, cs_db, d):

        #: (float) mixing energy, transition between hadron and resonance behavior
        self.E_mix = 0
        #: (int) energy grid index, where transition between hadron and resonance occurs
        self.mix_idx = 0
        #: (float) critical energy in air at the surface
        self.E_crit = 0

        #: (bool) particle is a hadron (meson or baryon)
        self.is_hadron = False
        #: (bool) particle is a meson
        self.is_meson = False
        #: (bool) particle is a baryon
        self.is_baryon = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.is_alias = False
        #: (bool) particle has both, hadron and resonance properties
        self.is_mixed = False
        #: (bool) if particle has just resonance behavior
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdgid = pdgid
        #: (int) MCEq ID
        self.nceidx = -1

        self.particle_db = particle_db
        self.pythia_db = pythia_db
        if pdgid in config["adv_set"]["disable_decays"]:
            pythia_db.force_stable(self.pdgid)
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
        return self.nceidx * self.d

    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.nceidx + 1) * self.d

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


class SpeciesManager(object):
    """Provides a database with particle and species."""

    def __init__(self, ncoid_list, ed):
        # (dict) Dimension of primary grid
        self.grid_dims = {'default': ed}
        # Particle index shortcuts
        #: (dict) Converts Neucosma ID to index in state vector
        self.ncoid2princeidx = {}
        #: (dict) Converts particle name to index in state vector
        self.sname2princeidx = {}
        #: (dict) Converts Neucosma ID to reference of
        # :class:`data.PrinceSpecies`
        self.ncoid2sref = {}
        #: (dict) Converts particle name to reference of
        #:class:`data.PrinceSpecies`
        self.sname2sref = {}
        #: (dict) Converts prince index to reference of
        #:class:`data.PrinceSpecies`
        self.princeidx2sref = {}
        #: (dict) Converts index in state vector to Neucosma ID
        self.princeidx2ncoid = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`data.PrinceSpecies`
        self.princeidx2pname = {}
        #: (int) Total number of species
        self.nspec = 0

        self._gen_species(ncoid_list)
        self._init_species_tables()

    def _gen_species(self, ncoid_list):
        info(4, "Generating list of species.")

        # ncoid_list += spec_data["non_nuclear_species"]

        # Make sure list is unique and sorted
        ncoid_list = sorted(list(set(ncoid_list)))

        self.species_refs = []
        # Define position in state vector (princeidx) by simply
        # incrementing it with the (sorted) list of Neucosma IDs
        for princeidx, ncoid in enumerate(ncoid_list):
            info(
                4, "Appending species {0} at position {1}".format(
                    ncoid, princeidx))
            self.species_refs.append(
                PrinceSpecies(ncoid, princeidx, self.grid_dims['default']))

        self.known_species = [s.ncoid for s in self.species_refs]
        self.redist_species = [
            s.ncoid for s in self.species_refs if s.has_redist
        ]
        self.boost_conserv_species = [
            s.ncoid for s in self.species_refs if not s.has_redist
        ]

    def _init_species_tables(self):
        for s in self.species_refs:
            self.ncoid2princeidx[s.ncoid] = s.princeidx
            self.sname2princeidx[s.sname] = s.princeidx
            self.princeidx2ncoid[s.princeidx] = s.ncoid
            self.princeidx2pname[s.princeidx] = s.sname
            self.ncoid2sref[s.ncoid] = s
            self.princeidx2sref[s.princeidx] = s
            self.sname2sref[s.sname] = s

        self.nspec = len(self.species_refs)

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
            str_out += 'NCO id : ' + str(s.ncoid) + '\n' + ident
            str_out += 'PriNCe idx : ' + str(s.princeidx) + '\n\n'

        return str_out