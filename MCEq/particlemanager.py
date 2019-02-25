# -*- coding: utf-8 -*-
"""
:mod:`MCEq.particlemanager` --- management of particle objects
==============================================================

This module includes code for bookkeeping, interfacing and
validating data structures:

- :class:`MCEqParticle` bundles different particle properties for simpler
  usage in :class:`MCEqRun`


"""
import six
import numpy as np
from mceq_config import config
from MCEq.misc import info, print_in_rows, getAZN

from particletools.tables import PYTHIAParticleData
info(5, 'Initialization of PYTHIAParticleData object')
_pdata = PYTHIAParticleData()

backward_compatible_namestr = {
    'nu_mu': 'numu',
    'nu_mubar': 'antinumu',
    'nu_e': 'nue',
    'nu_ebar': 'antinue',
    'nu_tau': 'nutau',
    'nu_taubar': 'antinutau'
}


# Replace particle names for neutrinos with those used in previous MCEq versions
def _pname(pdg_id_or_name):
    """Replace some particle names from pythia database with those from previous
    MCEq versions for backward compatibility."""

    pythia_name = _pdata.name(pdg_id_or_name)
    if pythia_name in backward_compatible_namestr:
        return backward_compatible_namestr[pythia_name]
    return pythia_name


class MCEqParticle(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdg_id (int): PDG ID of the particle
      egrid (np.array, optional): energy grid (centers)
      cs_db (object, optional): reference to an instance of :class:`InteractionYields`
    """

    def __init__(self,
                 pdg_id,
                 energy_grid=None,
                 cs_db=None,
                 init_pdata_defaults=True):

        #: (bool) if it's an electromagnetic particle
        self.is_em = abs(pdg_id) == 11 or pdg_id == 22
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = False
        #: (bool) particle is a hadron
        self.is_hadron = False
        #: (bool) particle is a hadron
        self.is_lepton = False
        #: (float) ctau in cm
        self.ctau = None
        #: (float) mass in GeV
        self.mass = None
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.name = None
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(pdg_id)
        #: (bool) particle has both, hadron and resonance properties
        self.is_alias = False
        #: (str) species name in string representation
        self.is_mixed = False
        #: (bool) if particle has just resonance behavior
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (bool) particle is stable
        self.is_stable = False
        #: (bool) can_interact
        self.can_interact = False
        #: decay channels if any
        self.decay_channels = {}
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdg_id = pdg_id
        #: (int) MCEq ID
        self.mceqidx = -1

        #: (float) mixing energy, transition between hadron and resonance behavior
        self.E_mix = 0
        #: (int) energy grid index, where transition between hadron and resonance occurs
        self.mix_idx = 0
        #: (float) critical energy in air at the surface
        self.E_crit = 0

        # # TODO: move this check to internal variable self.is_stable, or so
        # # if pdg_id in config["adv_set"]["disable_decays"]:
        # #     _pdata.force_stable(self.pdg_id)

        # Energy and cross section dependent inits
        self.current_cross_sections = None
        self._energy_grid = energy_grid

        # Variables for hadronic interaction
        self.current_hadronic_model = None
        self.hadr_secondaries = []
        self.hadr_yields = {}

        # Variables for decays
        self.children = []
        self.decay_dists = {}

        if init_pdata_defaults:
            self._init_defaults_from_pythia_database()

        if self._energy_grid is not None and cs_db is not None:
            #: interaction cross section in 1/cm2
            self.set_cs(cs_db)

    def _init_defaults_from_pythia_database(self):
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = _pdata.is_nucleus(self.pdg_id)
        #: (bool) particle is a hadron
        self.is_hadron = _pdata.is_hadron(self.pdg_id)
        #: (bool) particle is a hadron
        self.is_lepton = _pdata.is_lepton(self.pdg_id)
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(self.pdg_id)
        #: (float) ctau in cm
        self.ctau = _pdata.ctau(self.pdg_id)
        #: (float) mass in GeV
        self.mass = _pdata.mass(self.pdg_id)
        #: (str) species name in string representation
        self.name = _pname(self.pdg_id)
        #: (bool) particle is stable
        self.is_stable = not self.ctau < np.inf
    
    def init_custom_particle_data(self, name, pdg_id, ctau, mass, **kwargs):
        """Add custom particle type. (Incomplete)"""
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdg_id = pdg_id
        #: (bool) if it's an electromagnetic particle
        self.is_em = kwargs.pop('is_em', abs(pdg_id) == 11 or pdg_id == 22)
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = kwargs.pop('is_nucleus', _pdata.is_nucleus(self.pdg_id))
        #: (bool) particle is a hadron
        self.is_hadron = kwargs.pop('is_hadron', _pdata.is_hadron(self.pdg_id))
        #: (bool) particle is a hadron
        self.is_lepton = kwargs.pop('is_lepton', _pdata.is_lepton(self.pdg_id))
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(self.pdg_id)
        #: (float) ctau in cm
        self.ctau = ctau
        #: (float) mass in GeV
        self.mass = mass
        #: (str) species name in string representation
        self.name = name
        #: (bool) particle is stable
        self.is_stable = not self.ctau < np.inf

    def set_cs(self, cs_db):
        """Set cross section adn recalculate the dependent variables"""

        self.current_cross_sections = cs_db.iam
        self.cs = cs_db[self.pdg_id]
        if sum(self.cs) > 0:
            self.can_interact = True
        else:
            self.can_interact = False
        self._critical_energy()
        self._calculate_mixing_energy()

    def set_hadronic_channels(self, hadronic_db, pmanager):
        """Changes the hadronic interaction model.
        
        Replaces indexing of the yield dictionary from PDG IDs
        with references from partilcle manager.
        """

        self.current_hadronic_model = hadronic_db.iam

        if self.pdg_id in hadronic_db.projectiles:
            self.is_projectile = True
            self.hadr_secondaries = [
                pmanager.pdg2pref[pid]
                for pid in hadronic_db.secondary_dict[self.pdg_id]
                if abs(pid) < 7000
            ]
            self.hadr_yields = {}
            for s in self.hadr_secondaries:
                self.hadr_yields[s] = hadronic_db.yields[(self.pdg_id,
                                                          s.pdg_id)]
        else:
            self.is_projectile = False
            self.hadr_secondaries = []
            self.hadr_yields = {}

    def set_decay_channels(self, decay_db, pmanager):
        """Populates decay channel and energy distributions"""

        if self.is_stable:
            # Variables for decays
            self.children = []
            self.decay_dists = {}
            return

        if self.pdg_id not in decay_db.mothers:
            raise Exception('Unstable particle without decay distribution:',
                            self.pdg_id, self.name)

        self.children = []
        self.children = [pmanager[d] for d in decay_db.daughters(self.pdg_id)]
        self.decay_dists = {}
        for c in self.children:
            self.decay_dists[c] = decay_db.decay_dict[(self.pdg_id, c.pdg_id)]

    def is_secondary(self, pdg_id):
        """`True` if `self` is projectile and produces `pdg_id` particles."""
        return pdg_id in self.hadr_secondaries

    def is_child(self, pdg_id):
        """If particle decays into `pdg_id` return True."""
        return pdg_id in self.children

    @property
    def hadridx(self):
        """Returns index range where particle behaves as hadron.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (self.mix_idx, self._energy_grid.d)

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
        return self.mceqidx * self._energy_grid.d

    @property
    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.mceqidx + 1) * self._energy_grid.d

    def inverse_decay_length(self, cut=True):
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
            dlen = self.mass / self.ctau / self._energy_grid.c
            if cut:
                dlen[0:self.mix_idx] = 0.
            # TODO: verify how much this affects the result
            return 0.9966 * dlen  # Correction for bin average
        except ZeroDivisionError:
            return np.ones(self._energy_grid.d) * np.inf

    def inel_cross_section(self, mbarn=False):
        """Returns inverse interaction length for A_target given by config.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """
        #: unit - :math:`\text{GeV} \cdot \text{fm}`
        GeVfm = 0.19732696312541853
        #: unit - :math:`\text{GeV} \cdot \text{cm}`
        GeVcm = GeVfm * 1e-13
        #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
        GeV2mbarn = 10.0 * GeVfm**2
        #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
        mbarn2cm2 = GeVcm**2 / GeV2mbarn
        if mbarn:
            return mbarn2cm2 * self.cs

        return self.cs

    def inverse_interaction_length(self):
        """Returns inverse interaction length for A_target given by config.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_target = config['A_target'] * 1.672621 * 1e-24  # <A> * m_proton [g]
        return self.cs / m_target

    def get_xlab_dist(self, energy, sec_pdg, verbose=True, **kwargs):
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

        eidx = (np.abs(self._energy_grid.c - energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(2, 'Nearest energy, index: ', en, eidx, condition=verbose)

        m = self.hadr_yields[sec_pdg]
        xl_grid = self._energy_grid.c[:eidx + 1] / en
        xl_dist = xl_grid * en * m[:eidx + 1, eidx] / np.diag(
            self._energy_grid.w)[:eidx + 1]

        return xl_grid, xl_dist

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

        eidx = (np.abs(self._energy_grid.c - energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(2, 'Nearest energy, index: ', en, eidx, condition=verbose)
        m = self.hadr_yields[sec_pdg]
        xl_grid = self._energy_grid.c[:eidx + 1] / en
        xl_dist = xl_grid * en * m[:eidx + 1, eidx] / np.diag(
            self._energy_grid.w)[:eidx + 1]
        xf_grid, dxl_dxf = xF(xl_grid, en, sec_pdg)
        xf_dist = xl_dist * dxl_dxf

        if pos_only:
            xf_dist = xf_dist[xf_grid >= 0]
            xf_grid = xf_grid[xf_grid >= 0]
            return xf_grid, xf_dist

        return xf_grid, xf_dist

    def _critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.

        Returns:
          (float): :math:`\\frac{m\\ 6.4 \\text{km}}{c\\tau}` in GeV
        """
        try:
            self.E_crit = self.mass * 6.4e5 / self.ctau
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

        if abs(self.pdg_id) in [2212]:
            self.mix_idx = 0
            self.is_mixed = False
            return
        d = self._energy_grid.d

        inv_intlen = self.inverse_interaction_length()
        inv_declen = self.inverse_decay_length()
        if (not np.any(inv_declen > 0.) or abs(
                self.pdg_id) in config["adv_set"]["exclude_from_mixing"]):
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
            return

        if np.abs(self.pdg_id) in config["adv_set"]["force_resonance"]:
            threshold = 0.
        elif np.any(inv_intlen > 0.):
            lint = 1. / inv_intlen
            d_tilde = 1. / inv_declen
            # multiply with maximal density encountered along the
            # integration path
            ldec = d_tilde * max_density
            threshold = ldec / lint
        else:
            threshold = np.inf
            no_mix = True

        if np.max(threshold) < cross_over:
            self.mix_idx = d - 1
            self.E_mix = self._energy_grid.c[self.mix_idx]
            self.is_mixed = False
            self.is_resonance = True

        elif np.min(threshold) > cross_over or no_mix:
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
        else:
            self.mix_idx = np.where(ldec / lint > cross_over)[0][0]
            self.E_mix = self._energy_grid.c[self.mix_idx]
            self.is_mixed = True
            self.is_resonance = False

    def __repr__(self):
        a_string = ("""
        {0}:
        is_hadron   : {1}
        is_lepton   : {2}
        is_nucleus  : {3}
        is_mixed    : {4}
        is_resonance: {5}
        is_alias    : {6}
        E_mix       : {7:2.1e}\n""").format(
            self.name, self.is_hadron, self.is_lepton, self.is_nucleus,
            self.is_mixed, self.is_resonance, self.is_alias, self.E_mix)
        return a_string


class ParticleManager(object):
    """Database for objects of :class:`MCEqParticle`.
    
    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """

    def __init__(self, pdg_id_list, energy_grid, cs_db, mod_table=None):
        # (dict) Dimension of primary grid
        self._energy_grid = energy_grid
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

        # Cross section database
        self._cs_db = cs_db

        self._init_categories(particle_pdg_list=pdg_id_list)

        self._update_particle_tables()

        self.print_particle_tables(2)

    def _update_particle_tables(self):
        """Update internal mapping tables after changes to the particle
        list occur."""

        self.n_cparticles = len(self.cascade_particles)
        self.dim = self._energy_grid.d
        self.dim_states = self._energy_grid.d * self.n_cparticles
        self.muon_selector = np.zeros(self.dim_states, dtype='bool')

        for p in self.all_particles:
            self.pdg2mceqidx[p.pdg_id] = p.mceqidx
            self.pname2mceqidx[p.name] = p.mceqidx
            self.mceqidx2pdg[p.mceqidx] = p.pdg_id
            self.mceqidx2pname[p.mceqidx] = p.name
            self.mceqidx2pref[p.mceqidx] = p
            self.pdg2pref[p.pdg_id] = p
            self.pname2pref[p.name] = p

            # TODO: This thing has to change to something like
            # "partiles with continuous losses"
            # Select all positions of muon species in the state vector
            if abs(p.pdg_id) == 13:
                self.muon_selector[p.lidx:p.uidx] = True

        # self._init_alias_tables()
        # self._init_muon_energy_loss()

        self.print_particle_tables(10)

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
        else:
            from particletools.tables import SibyllParticleTable
            modtab = SibyllParticleTable()
            particles = modtab.baryons + modtab.mesons + modtab.leptons

        # TODO: hotfix to remove the special categories
        particles = [p for p in particles if abs(p) < 7000]

        # Remove duplicates
        particles = sorted(list(set(particles)))

        # Initialize particle objects
        particle_list = [
            MCEqParticle(p, self._energy_grid, self._cs_db) for p in particles
        ]

        # Sort by critical energy (= int_len ~== dec_length ~ int_cs/tau)
        particle_list.sort(key=lambda x: x.E_crit, reverse=False)

        # Cascade particles will "live" on the grid and have an mceqidx assigned
        self.cascade_particles = [
            p for p in particle_list if not p.is_resonance
        ]

        # These particles will only exist implicitely and integated out
        self.resonances = [p for p in particle_list if p.is_resonance]

        # Assign an mceqidx (position in state vector) to each explicit particle
        # Resonances will kepp the default mceqidx = -1
        for mceqidx, h in enumerate(self.cascade_particles):
            h.mceqidx = mceqidx

        self.all_particles = self.cascade_particles + self.resonances

    def add_tracking_particle(self, parent_list, child_pdg, alias_name):
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

        particle = MCEqParticle(child_pdg, self._energy_grid, self._cs_db)
        particle.is_alias = True

    def set_cross_sections_db(self, cs_db):
        """Let all particles know about their inelastic cross section"""
        info(5, 'Setting cross section particle variables.')

        for p in self.cascade_particles:
            if p.current_cross_sections == cs_db.iam:
                continue
            p.set_cs(cs_db)

        self._update_particle_tables()

    def set_decay_channels(self, decay_db):
        info(5, 'Setting decay info for particles.')
        for p in self.all_particles:
            p.set_decay_channels(decay_db, self)
        self._update_particle_tables()

    def set_interaction_model(self, cs_db, hadronic_db):
        info(5, 'Setting hadronic secondaries for particles.')
        for p in self.cascade_particles:
            if p.current_cross_sections != cs_db.iam:
                p.set_cs(cs_db)
            if p.current_hadronic_model != hadronic_db.iam:
                p.set_hadronic_channels(hadronic_db, self)

        self._update_particle_tables()

    def __getitem__(self, pdg_id_or_name):
        """Returns reference to particle object."""
        if isinstance(pdg_id_or_name, six.integer_types):
            return self.pdg2pref[pdg_id_or_name]
        else:
            return self.pdg2pref[_pname(pdg_id_or_name)]

    def keys(self):
        """Returns pdg_ids of all particles"""
        return [p.pdg_id for p in self.all_particles]

    # def _init_alias_tables(self):
    #     r"""Sets up the functionality of aliases and defines the meaning of
    #     'prompt'.

    #     The identification of the last mother particle of a lepton is implemented
    #     via index aliases. I.e. the PDG index of muon neutrino 14 is transformed
    #     into 7114 if it originates from decays of a pion, 7214 in case of kaon or
    #     7014 if the mother particle is very short lived (prompt). The 'very short lived'
    #     means that the critical energy :math:`\varepsilon \ge \varepsilon(D^\pm)`.
    #     This includes all charmed hadrons, as well as resonances such as :math:`\eta`.

    #     The aliases for the special ``obs_`` category are also initialized here.
    #     """
    #     info(5, "Initializing links to alias IDs.")

    #     self.alias_table = {}
    #     prompt_ids = []
    #     for p in self.all_particles:
    #         if p.is_lepton or p.is_alias or p.pdg_id < 0:
    #             continue
    #         if 411 in self.pdg2pref and p.E_crit >= self.pdg2pref[411].E_crit:
    #             prompt_ids.append(p.pdg_id)
    #     for lep_id in [12, 13, 14, 16]:
    #         self.alias_table[(211, lep_id)] = 7100 + lep_id  # pions
    #         self.alias_table[(321, lep_id)] = 7200 + lep_id  # kaons
    #         for pr_id in prompt_ids:
    #             self.alias_table[(pr_id, lep_id)] = 7000 + lep_id  # prompt

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
            str_out += 'PDG id : ' + str(s.pdg_id) + '\n' + ident
            str_out += 'MCEq idx : ' + str(s.mceqidx) + '\n\n'

        return str_out

    def print_particle_tables(self, min_dbg_lev=2):

        info(min_dbg_lev, "\nHadrons and stable particles:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name for p in self.all_particles
            if p.is_hadron and not p.is_resonance and not p.is_mixed
        ])

        info(min_dbg_lev, "\nMixed:\n", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_mixed])

        info(min_dbg_lev, "\nResonances:\n", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_resonance])

        info(min_dbg_lev, "\nLeptons:\n", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name
            for p in self.all_particles if p.is_lepton and not p.is_alias
        ])
        info(min_dbg_lev, "\nAliases:\n", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_alias])

        info(
            min_dbg_lev,
            "\nTotal number of species:",
            self.n_cparticles,
            no_caller=True)

        # list particle indices
        if False:
            info(10, "Particle matrix indices:", no_caller=True)
            some_index = 0
            for p in self.cascade_particles:
                for i in xrange(self._energy_grid.d):
                    info(10, p.name + '_' + str(i), some_index, no_caller=True)
                    some_index += 1