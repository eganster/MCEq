# -*- coding: utf-8 -*-
"""
:mod:`MCEq.core` - core module
==============================

This module contains the main program features. Instantiating :class:`MCEq.core.MCEqRun`
will initialize the data structures and particle tables, create and fill the
interaction and decay matrix and check if all information for the calculation
of inclusive fluxes in the atmosphere is available.

The preferred way to instantiate :class:`MCEq.core.MCEqRun` is::

    from mceq_config import config
    from MCEq.core import MCEqRun
    import CRFluxModels as pm

    mceq_run = MCEqRun(interaction_model='SIBYLL2.3c',
                       primary_model=(pm.HillasGaisser2012, "H3a"),
                       **config)

    mceq_run.set_theta_deg(60.)
    mceq_run.solve()

"""
from time import time
import numpy as np
from mceq_config import config
from MCEq.misc import normalize_hadronic_model_name, info, energy_grid
from MCEq.particlemanager import ParticleManager


class MCEqRun(object):
    """Main class for handling the calculation.

    This class is the main user interface for the caclulation. It will
    handle initialization and various error/configuration checks. The
    setup has to be accomplished before invoking the integration routine
    is :func:`MCeqRun.solve`. Changes of configuration, such as:

    - interaction model in :meth:`MCEqRun.set_interaction_model`,
    - primary flux in :func:`MCEqRun.set_primary_model`,
    - zenith angle in :func:`MCEqRun.set_theta_deg`,
    - density profile in :func:`MCEqRun.set_density_model`,
    - member particles of the special ``obs_`` group in :func:`MCEqRun.set_obs_particles`,

    can be made on an active instance of this class, while calling
    :func:`MCEqRun.solve` subsequently to calculate the solution
    corresponding to the settings.

    The result can be retrieved by calling :func:`MCEqRun.get_solution`.


    Args:
      interaction_model (string): PDG ID of the particle
      density_model (string,sting,string): model type, location, season
      primary_model (class, param_tuple): classes derived from
        :class:`CRFluxModels.PrimaryFlux` and its parameters as tuple
      theta_deg (float): zenith angle :math:`\\theta` in degrees,
        measured positively from vertical direction
      adv_set (dict): advanced settings, see :mod:`mceq_config`
      obs_ids (list): list of particle name strings. Those lepton decay
        products will be scored in the special ``obs_`` categories
    """

    def __init__(self, interaction_model, density_model, primary_model,
                 theta_deg, adv_set, obs_ids, **kwargs):

        # from particletools.tables import SibyllParticleTable, PYTHIAParticleData
        from MCEq.data import DecayYields, InteractionYields, HadAirCrossSections

        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_config = density_model
        self.theta_deg = theta_deg

        # Load particle production yields
        self.yields_params = dict(interaction_model=interaction_model, charm_model=None)
        #: handler for decay yield data of type :class:`MCEq.data.InteractionYields`
        self.y = InteractionYields(**self.yields_params)
        # Interaction matrices initialization flag
        self.iam_mat_initialized = False
        # Load decay spectra
        self.decays_params = dict(mother_list=self.y.particle_list, )

        #: handler for decay yield data of type :class:`MCEq.data.DecayYields`
        self.decays = DecayYields(**self.decays_params)

        # Load cross-section handling
        self.cs_params = dict(interaction_model=interaction_model)
        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self.cs = HadAirCrossSections(**self.cs_params)

        # Save primary model params
        self.pm_params = primary_model

        # Store adv_set
        self.adv_set = adv_set

        # First interaction mode
        self.fa_vars = None

        # Default GPU device id for CUDA
        self.cuda_device = kwargs['GPU_id'] if 'GPU_id' in kwargs else 0

        # Store array precision (TODO: obsolete this for non GPU runs)
        if config['FP_precision'] == 32:
            self.fl_pr = np.float32
        elif config['FP_precision'] == 64:
            self.fl_pr = np.float64
        else:
            raise Exception("MCEqRun(): Unknown float precision specified.")

        # General Matrix dimensions and shortcuts, controlled by
        # grid of yield matrices
        self._energy_grid = energy_grid(self.y.e_grid, self.y.e_bins,
                                        self.y.e_bins[1:] - self.y.e_bins[:-1],
                                        self.y.e_grid.size)

        # Custom particle list can be defined
        particle_list = kwargs.pop(
            'particle_list', self.y.particle_list + self.decays.particle_list)

        # Create particle database
        self.pman = ParticleManager(particle_list, self._energy_grid, self.cs)
        # TODO: this doesn't belong here
        self.pman.set_decay_channels(self.decays)

        # Set id of particles in observer category
        # self.pman.set_obs_particles(obs_ids)

        # Obtain total number of "cascade" particles and dimension of grid
        self.dim_states = self.pman.dim_states

        self.matrix_builder = MatrixBuilder(self.pman, self.y, self.decays)

        # Initialize solution vector
        self.solution = np.zeros(self.dim_states)

        # Initialize empty state (particle density) vector
        self.phi0 = np.zeros(self.dim_states)

        # Initialize muon energy loss
        self._init_muon_energy_loss()

        # Set interaction model and compute grids and matrices
        self.set_interaction_model(interaction_model, force=True)
        
        # Set atmosphere and geometry TODO do not allow empty density model
        # if density_model is not None:
        self.set_density_model(self.density_config)

        # Set initial flux condition
        if primary_model is not None:
            self.set_primary_model(*self.pm_params)

    @property
    def e_grid(self):
        """Energy grid (bin centers)"""
        return self._energy_grid.c

    @property
    def e_bins(self):
        """Energy grid (bin edges)"""
        return self._energy_grid.b

    @property
    def e_widths(self):
        """Energy grid (bin widths)"""
        return self._energy_grid.w
    
    @property
    def dim(self):
        """Energy grid (dimension)"""
        return self._energy_grid.d 

    

    # def _alias(self, mother, daughter):
    #     """Returns pair of alias indices, if ``mother``/``daughter`` combination
    #     belongs to an alias.

    #     Args:
    #       mother (int): PDG ID of mother particle
    #       daughter (int): PDG ID of daughter particle
    #     Returns:
    #       tuple(int, int): lower and upper index in state vector of alias or ``None``
    #     """

    #     ref = self.pdg2pref
    #     abs_mo = np.abs(mother)
    #     abs_d = np.abs(daughter)
    #     si_d = np.sign(daughter)
    #     if (abs_mo, abs_d) in self.alias_table.keys():
    #         return (ref[si_d * self.alias_table[(abs_mo, abs_d)]].lidx,
    #                 ref[si_d * self.alias_table[(abs_mo, abs_d)]].uidx)
    #     else:
    #         return None

    # def _alternate_score(self, mother, daughter):
    #     """Returns pair of special score indices, if ``mother``/``daughter`` combination
    #     belongs to the ``obs_`` category.

    #     Args:
    #       mother (int): PDG ID of mother particle
    #       daughter (int): PDG ID of daughter particle
    #     Returns:
    #       tuple(int, int): lower and upper index in state vector of ``obs_`` group or ``None``
    #     """

    #     info(20, 'Look-up for particles to score in `obs`', mother, daughter)

    #     ref = self.pdg2pref
    #     abs_mo = np.abs(mother)
    #     abs_d = np.abs(daughter)
    #     si_d = np.sign(daughter)
    #     if (abs_mo, abs_d) in self.obs_table.keys():
    #         return (ref[si_d * self.obs_table[(abs_mo, abs_d)]].lidx,
    #                 ref[si_d * self.obs_table[(abs_mo, abs_d)]].uidx)
    #     else:
    #         return None

    def get_solution(self,
                     particle_name,
                     mag=0.,
                     grid_idx=None,
                     integrate=False):
        """Retrieves solution of the calculation on the energy grid.

        Some special prefixes are accepted for lepton names:

        - the total flux of muons, muon neutrinos etc. from all sources/mothers
          can be retrieved by the prefix ``total_``, i.e. ``total_numu``
        - the conventional flux of muons, muon neutrinos etc. from all sources
          can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
        - correspondigly, the flux of leptons which originated from the decay
          of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``
        - conventional leptons originating neither from pion nor from kaon
          decay are collected in a category without any prefix, e.g. ``numu`` or
          ``mu+``

        Args:
          particle_name (str): The name of the particle such, e.g.
            ``total_mu+`` for the total flux spectrum of positive muons or
            ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
          mag (float, optional): 'magnification factor': the solution is
            multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
          grid_idx (int, optional): if the integrator has been configured to save
            intermediate solutions on a depth grid, then ``grid_idx`` specifies
            the index of the depth grid for which the solution is retrieved. If
            not specified the flux at the surface is returned
          integrate (bool, optional): return averge particle number instead of
          flux (multiply by bin width)

        Returns:
          (numpy.array): flux of particles on energy grid :attr:`e_grid`
        """
        # Account for the

        res = np.zeros(self._energy_grid.d)
        ref = self.pman
        sol = None
        if grid_idx is None:
            sol = self.solution
        elif grid_idx >= len(self.grid_sol):
            sol = self.grid_sol[-1]
        else:
            sol = self.grid_sol[grid_idx]

        if particle_name.startswith('total'):
            lep_str = particle_name.split('_')[1]
            for prefix in ('pr_', 'pi_', 'k_', ''):
                particle_name = prefix + lep_str
                res += sol[ref[particle_name].lidx:
                           ref[particle_name].uidx] * \
                    self._energy_grid.c ** mag
        elif particle_name.startswith('conv'):
            lep_str = particle_name.split('_')[1]
            for prefix in ('pi_', 'k_', ''):
                particle_name = prefix + lep_str
                res += sol[ref[particle_name].lidx:
                           ref[particle_name].uidx] * \
                    self._energy_grid.c ** mag
        else:
            print particle_name, ref.pname2pref[particle_name].lidx, ref.pname2pref[particle_name].uidx
            res = sol[ref.pname2pref[particle_name].lidx:
                      ref.pname2pref[particle_name].uidx] * \
                self._energy_grid.c ** mag
        if not integrate:
            return res
        else:
            return res * self._energy_grid.w

    # def set_obs_particles(self, obs_ids):
    #     """Adds a list of mother particle strings which decay products
    #     should be scored in the special ``obs_`` category.

    #     Decay and interaction matrix will be regenerated automatically
    #     after performing this call.

    #     Args:
    #       obs_ids (list of strings): mother particle names
    #     """

    #     #TODO: Forward this to ParticleManager
    #     if obs_ids is None:
    #         self.obs_ids = None
    #         return
    #     self.obs_ids = []
    #     for obs_id in obs_ids:
    #         try:
    #             self.obs_ids.append(int(obs_id))
    #         except ValueError:
    #             self.obs_ids.append(self.modtab.modname2pdg[obs_id])
    #     info(5, 'Converted names:', ', '.join([str(oid) for oid in obs_ids]),
    #          '\nto:', ', '.join([str(oid) for oid in self.obs_ids]))

    # self._init_alias_tables()
        # self.int_m, self.dec_m = self.matrix_builder.construct_matrices(skip_decay_matrix=False)

    def set_interaction_model(self,
                              interaction_model,
                              charm_model=None,
                              force=False):
        """Sets interaction model and/or an external charm model for calculation.

        Decay and interaction matrix will be regenerated automatically
        after performing this call.

        Args:
          interaction_model (str): name of interaction model
          charm_model (str, optional): name of charm model
          force (bool): force loading interaction model
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

        info(1, interaction_model)

        if not force and (
            (self.yields_params['interaction_model'],
             self.yields_params['charm_model']) == (interaction_model,
                                                    charm_model)):
            info(2, 'Skip, since current model identical to',
                 interaction_model + '/' + str(charm_model) + '.')
            return

        self.yields_params['interaction_model'] = interaction_model
        self.yields_params['charm_model'] = charm_model

        self.y.set_interaction_model(interaction_model)
        self.y._inject_custom_charm_model(charm_model)

        self.cs_params['interaction_model'] = interaction_model
        self.cs.set_interaction_model(interaction_model)

        self.pman.set_interaction_model(self.cs, self.y)

        # initialize matrices
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(skip_decay_matrix=False)

    def set_primary_model(self, mclass, tag):
        """Sets primary flux model.

        This functions is quick and does not require re-generation of
        matrices.

        Args:
          interaction_model (:class:`CRFluxModel.PrimaryFlux`): reference
          to primary model **class**
          tag (tuple): positional argument list for model class
        """

        info(1, mclass.__name__, tag)

        # Initialize primary model object
        self.pmodel = mclass(tag)
        self.get_nucleon_spectrum = np.vectorize(self.pmodel.p_and_n_flux)

        try:
            self.dim_states
        except AttributeError:
            self.finalize_pmodel = True

        # Save initial condition
        self.phi0 *= 0
        p_top, n_top = self.get_nucleon_spectrum(self._energy_grid.c)[1:]
        self.phi0[self.pman[2212].lidx:self.pman[2212].uidx] = 1e-4 * p_top

        if 2112 in self.pman.keys() and not self.pman[2112].is_resonance:
            self.phi0[self.pman[2112].lidx:self.pman[2112].
                      uidx] = 1e-4 * n_top
        else:
            self.phi0[self.pman[2212].lidx:self.pman[2212].
                      uidx] += 1e-4 * n_top

    def set_single_primary_particle(self, E, corsika_id):
        """Set type and energy of a single primary nucleus to
        calculation of particle yields.

        The functions uses the superposition theorem, where the flux of
        a nucleus with mass A and charge Z is modeled by using Z protons
        and A-Z neutrons at energy :math:`E_{nucleon}= E_{nucleus} / A`
        The nucleus type is defined via :math:`\\text{CORSIKA ID} = A*100 + Z`. For
        example iron has the CORSIKA ID 5226.

        A continuous input energy range is allowed between
        :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.

        Args:
          E (float): (total) energy of nucleus in GeV
          corsika_id (int): ID of nucleus (see text)
        """

        from scipy.linalg import solve
        from MCEq.misc import getAZN_corsika
        
        info(
            2, 'corsika_id={0}, particle energy={1:5.3g} GeV'.format(
                corsika_id, E))

        egrid = self._energy_grid.c
        ebins = self._energy_grid.b
        ewidths = self._energy_grid.w

        n_nucleons, n_protons, n_neutrons = getAZN_corsika(corsika_id)
        En = E/float(n_nucleons)
        
        if En < np.min(self._energy_grid.c):
            raise Exception('energy per nucleon too low for primary ' +
                            str(corsika_id))

        info(3, ('superposition: n_protons={0}, n_neutrons={1}, ' +
                 'energy per nucleon={2:5.3g} GeV').format(
                     n_protons, n_neutrons, En))

        cenbin = np.argwhere(En < ebins)[0][0] - 1

        # Equalize the first three moments for 3 normalizations around the central
        # bin
        emat = np.vstack(
            (ewidths[cenbin - 1:cenbin + 2],
             ewidths[cenbin - 1:cenbin + 2] * egrid[cenbin - 1:cenbin + 2],
             ewidths[cenbin - 1:cenbin + 2] * egrid[cenbin - 1:cenbin + 2]**2))

        b_protons = np.array([n_protons, En * n_protons, En**2 * n_protons])
        b_neutrons = np.array(
            [n_neutrons, En * n_neutrons, En**2 * n_neutrons])

        self.phi0 *= 0. 

        if n_protons > 0:
            p_lidx = self.pman[2212].lidx
            self.phi0[p_lidx + cenbin - 1:p_lidx + cenbin + 2] = solve(
                emat, b_protons)
        if n_neutrons > 0:
            n_lidx = self.pman[2112].lidx
            self.phi0[n_lidx + cenbin - 1:n_lidx + cenbin + 2] = solve(
                emat, b_neutrons)

    def set_density_model(self, density_config):
        """Sets model of the atmosphere.

        To choose, for example, a CORSIKA parametrization for the Southpole in January,
        do the following::

            mceq_instance.set_density_model(('CORSIKA', 'PL_SouthPole', 'January'))

        More details about the choices can be found in :mod:`MCEq.geometry.density_profiles`. Calling
        this method will issue a recalculation of the interpolation and the integration path.

        Args:
          density_config (tuple of strings): (parametrization type, arguments)
        """
        import MCEq.geometry.density_profiles as dprof

        base_model, model_config = density_config

        available_models = ['MSIS00', 'MSIS00_IC', 'CORSIKA', 'AIRS', 'Isothermal', 'GeneralizedTarget']
        info(1, 'Setting density profile to', base_model, model_config)
        if base_model not in available_models:
            info(0, 'Unknown density model. Available choices are:\n', '\n'.join(available_models))
            raise Exception('Choose a different profile.')

        info(1, 'Setting density profile to', base_model, model_config)

        if base_model == 'MSIS00':
            self.density_model = dprof.MSIS00Atmosphere(*model_config)
        elif base_model == 'MSIS00_IC':
            self.density_model = dprof.MSIS00IceCubeCentered(*model_config)
        elif base_model == 'CORSIKA':
            self.density_model = dprof.CorsikaAtmosphere(*model_config)
        elif base_model == 'AIRS':
            self.density_model = dprof.AIRSAtmosphere(*model_config)
        elif base_model == 'Isothermal':
            self.density_model = dprof.IsothermalAtmosphere(*model_config)
        elif base_model == 'GeneralizedTarget':
            self.density_model = dprof.GeneralizedTarget()
        else:
            raise Exception(
                'Unknown atmospheric base model.'
            )
        self.density_config = density_config

        if self.theta_deg is not None and base_model != 'GeneralizedTarget':
            self.set_theta_deg(self.theta_deg)
        elif base_model == 'GeneralizedTarget':
            self.integration_path = None

        # TODO: Make the pman aware of that density might have changed and
        # indices as well
        # self.pmod._gen_list_of_particles()

    def set_theta_deg(self, theta_deg):
        """Sets zenith angle :math:`\\theta` as seen from a detector.

        Currently only 'down-going' angles (0-90 degrees) are supported.

        Args:
          atm_config (tuple of strings): (parametrization type, location string, season string)
        """
        info(1, theta_deg)

        if self.density_config[0] == 'GeneralizedTarget':
            raise Exception('GeneralizedTarget does not support angles.')

        if self.density_model.theta_deg == theta_deg:
            info(2,
                 'Theta selection correponds to cached value, skipping calc.')
            return

        self.density_model.set_theta(theta_deg)
        self.integration_path = None

    def set_mod_pprod(self,
                      prim_pdg,
                      sec_pdg,
                      x_func,
                      x_func_args,
                      delay_init=False):
        """Sets combination of projectile/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of
        ``prim_pdg`` is modified according to the function passed to
        :func:`InteractionYields.init_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
          x_func (object): reference to function
          x_func_args (tuple): arguments passed to ``x_func``
          delay_init (bool): Prevent init of mceq matrices if you are
                             planning to add more modifications
        """
        # TODO: Debug this
        info(
            1, '{0}/{1}, {2}, {3}'.format(prim_pdg, sec_pdg, x_func.__name__,
                                          str(x_func_args)))

        init = self.y._set_mod_pprod(prim_pdg, sec_pdg, x_func, x_func_args)

        # Need to regenerate matrices completely
        return int(init)

        if init and not delay_init:
            self.int_m, self.dec_m = self.matrix_builder.construct_matrices(skip_decay_matrix=True)
            return 0

    def unset_mod_pprod(self, dont_fill=False):
        """Removes modifications from :func:`MCEqRun.set_mod_pprod`.

        Args:
          skip_fill (bool): If `true` do not regenerate matrices
          (has to be done at a later step by hand)
        """
        # TODO: Debug this
        from collections import defaultdict
        info(1, 'Particle production modifications reset to defaults.')

        self.y.mod_pprod = defaultdict(lambda: {})
        # Need to regenerate matrices completely
        if not dont_fill:
            self.int_m, self.dec_m = self.matrix_builder.construct_matrices(skip_decay_matrix=True)

    def solve(self, **kwargs):
        """Launches the solver.

        The setting `integrator` in the config file decides which solver
        to launch, either the simple but accelerated explicit Euler solvers, 
        :func:`MCEqRun._forward_euler` or, solvers from ODEPACK
        :func:`MCEqRun._odepack`.

        Args:
          kwargs (dict): Arguments are passed directly to the solver methods.

        """
        info(
            2, "solver={0} and sparse={1}".format(config['integrator'],
                                                  config['use_sparse']))

        if config['integrator'] != "odepack":
            self._forward_euler(**kwargs)
        elif config['integrator'] == 'odepack':
            self._odepack(**kwargs)
        else:
            raise Exception("Unknown integrator selection '{0}'.".format(
                config['integrator']))

    def _init_muon_energy_loss(self):
        # Muon energy loss
        import cPickle as pickle
        from os.path import join
        if config['energy_solver'] != 'Semi-Lagrangian':
            eloss_fname = str(config['mu_eloss_fname'][:-4] + '_centers.ppl')
        else:
            eloss_fname = str(config['mu_eloss_fname'][:-4] + '_edges.ppl')
        self.mu_dEdX = pickle.load(
            open(join(config['data_dir'], eloss_fname), 'rb')).astype(
                self.fl_pr) * 1e-3  # ... to GeV

        # Left index of first muon species and number of
        # muon species including aliases
        self.mu_loss_handler = None

    def _forward_euler(self, int_grid=None, grid_var='X'):
        """Solves the transport equations with solvers from :mod:`MCEq.time_solvers`.

        Args:
          int_grid (list): list of depths at which results are recorded
          grid_var (str): Can be depth `X` or something else (currently only `X` supported)

        """

        # Calculate integration path if not yet happened
        self._calculate_integration_path(int_grid, grid_var)

        phi0 = np.copy(self.phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, 'Solver will perform {0} integration steps.'.format(nsteps))

        import MCEq.time_solvers as time_solvers
        import MCEq.energy_solvers as energy_solvers

        if config["enable_muon_energy_loss"] and self.mu_loss_handler is None:
            # TODO: Make all solvers work with just muon selector
            if config["energy_solver"] == 'Semi-Lagrangian':
                self.mu_loss_handler = energy_solvers.SemiLagrangian(
                    self._e_bins, self.mu_dEdX,
                    self.muon_selector)
            elif config["energy_solver"] == 'Chang-Cooper':
                self.mu_loss_handler = energy_solvers.ChangCooper(
                    self._e_bins, self.mu_dEdX,
                    self.muon_selector)
            elif config["energy_solver"] == 'Direct':
                self.mu_loss_handler = energy_solvers.DifferentialOperator(
                    self.e_bins, self.mu_dEdX,
                    self.pman.muon_selector)
            else:
                raise Exception('Unknown energy solver.')

        start = time()

        if (config['first_interaction_mode']
                and config['kernel_config'] != 'numpy'):
            raise Exception("First interaction mode only works with numpy.")

        if config['kernel_config'] == 'numpy':
            kernel = time_solvers.solv_numpy
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0,
                    grid_idcs, self.mu_loss_handler, self.fa_vars)
        elif (config['kernel_config'] == 'CUDA'
              and config['use_sparse'] is True):
            kernel = time_solvers.solv_CUDA_sparse
            try:
                self.cuda_context.set_matrices(self.int_m, self.dec_m)
            except AttributeError:
                from MCEq.time_solvers import CUDASparseContext
                self.cuda_context = CUDASparseContext(
                    self.int_m, self.dec_m, device_id=self.cuda_device)
            args = (nsteps, dX, rho_inv, self.cuda_context, phi0, grid_idcs,
                    self.mu_loss_handler)

        elif (config['kernel_config'] == 'MKL'
              and config['use_sparse'] is True):
            kernel = time_solvers.solv_MKL_sparse
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0,
                    grid_idcs, self.mu_loss_handler)

        else:
            raise Exception(
                "Unsupported integrator settings '{0}/{1}'.".format(
                    'sparse' if config['use_sparse'] else 'dense',
                    config['kernel_config']))

        self.solution, self.grid_sol = kernel(*args)

        info(2,
             'time elapsed during integration: {0} sec'.format(time() - start))

    def _calculate_integration_path(self, int_grid, grid_var, force=False):

        if (self.integration_path and np.alltrue(int_grid == self.int_grid)
                and np.alltrue(self.grid_var == grid_var) and not force):
            info(5, 'skipping calculation.')
            return

        self.int_grid, self.grid_var = int_grid, grid_var
        if grid_var != 'X':
            raise NotImplementedError(
                'the choice of grid variable other than the depth X are not possible, yet.'
            )

        max_X = self.density_model.max_X
        ri = self.density_model.r_X2rho
        max_ldec = self.matrix_builder.max_ldec
        max_lint = self.matrix_builder.max_lint
        info(2, 'X_surface = {0}'.format(max_X))

        dX_vec = []
        rho_inv_vec = []

        X = 0.0
        step = 0
        grid_step = 0
        grid_idcs = []
        fa_vars = {}
        if config['first_interaction_mode']:
            # Create variables for first interaction mode

            old_err_state = np.seterr(divide='ignore')
            # TODO: this doesn't work anymore, fix
            fa_vars['Lambda_int'] = self.Lambda_int
            fa_vars['lint'] = 1 / self.Lambda_int
            np.seterr(**old_err_state)

            fa_vars['ipl'] = self.pman[2212].lidx
            fa_vars['ipu'] = self.pman[2212].uidx
            fa_vars['inl'] = self.pman[2112].lidx
            fa_vars['inu'] = self.pman[2112].uidx

            fa_vars[
                'pint'] = 1 / self.Lambda_int[fa_vars['ipl']:fa_vars['ipu']]
            fa_vars[
                'nint'] = 1 / self.Lambda_int[fa_vars['inl']:fa_vars['inu']]
            # Where to terminate the switch vector
            # (after all particle prod. for all species is disabled)
            fa_vars['max_step'] = 0

            fa_vars['fi_switch'] = []

        # The factor 0.95 means 5% inbound from stability margin of the
        # Euler intergrator.
        if (max_ldec * ri(config['max_density']) > max_lint
                and config["leading_process"] == 'decays'):
            info(2, "using decays as leading eigenvalues")
            delta_X = lambda X: 0.95 / (max_ldec * ri(X))
        else:
            info(2, "using interactions as leading eigenvalues")
            delta_X = lambda X: 0.95 / max_lint

        while X < max_X:
            dX = delta_X(X)
            if (np.any(int_grid) and (grid_step < len(int_grid))
                    and (X + dX >= int_grid[grid_step])):
                dX = int_grid[grid_step] - X
                grid_idcs.append(step)
                grid_step += 1
            dX_vec.append(dX)
            rho_inv_vec.append(ri(X))

            if config['first_interaction_mode'] and fa_vars['max_step'] == 0:
                # Only protons and neutrons can produce particles
                # Disable all interactions
                fa_vars['fi_switch'].append(
                    np.zeros(self.dim_states, dtype='double'))
                # Switch on particle production only for X < 1 x lambda_int
                fa_vars['fi_switch'][-1][fa_vars['ipl']:fa_vars['ipu']][
                    fa_vars['pint'] > X] = 1.
                fa_vars['fi_switch'][-1][fa_vars['inl']:fa_vars['inu']][
                    fa_vars['nint'] > X] = 1.
                # Save step value after which no particles are produced anymore
                if not np.sum(fa_vars['fi_switch'][-1]) > 0:
                    fa_vars['max_step'] = step
                # Promote fa_vars to class attribute
                self.fa_vars = fa_vars

            X = X + dX
            step += 1

        # Integrate
        dX_vec = np.array(dX_vec)
        rho_inv_vec = np.array(rho_inv_vec)

        self.integration_path = len(dX_vec), dX_vec, rho_inv_vec, grid_idcs

class MatrixBuilder(object):
    """This class constructs the interaction and decay matrices."""
    def __init__(self, particle_manager, yields, decays):
        self.pman = particle_manager
        self.dim_states = self.pman.dim_states
        self.dim = self.pman.dim
        self.y = yields
        self.decays = decays
        self.int_m = None
        self.dec_m = None
    
    def construct_matrices(self, skip_decay_matrix=False):
        r"""Constructs the matrices for calculation.

        These are:

        - :math:`\boldsymbol{M}_{int} = (-\boldsymbol{1} + \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}`,
        - :math:`\boldsymbol{M}_{dec} = (-\boldsymbol{1} + \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}`.

        For debug_levels >= 2 some general information about matrix shape and the number of
        non-zero elements is printed. The intermediate matrices :math:`\boldsymbol{C}` and
        :math:`\boldsymbol{D}` are deleted afterwards to save memory.

        Set the ``skip_decay_matrix`` flag to avoid recreating the decay matrix. This is not necessary
        if, for example, particle production is modified, or the interaction model is changed.

        Args:
          skip_decay_matrix (bool): Omit re-creating D matrix

        """

        # from scipy.sparse import bsr_matrix, coo_matrix
        from itertools import product
        info(2, "Start filling matrices. Skip_decay_matrix = {0}".format(
                skip_decay_matrix))

        self._fill_matrices(skip_decay_matrix=skip_decay_matrix)

        cparts = self.pman.cascade_particles

        # interaction part
        # -I + C
        # In first interaction mode it is just C
        self.max_lint = 0.
        for parent, child in product(cparts,cparts):
            idx = (child.mceqidx, parent.mceqidx)
            # Main diagonal
            if ((child.mceqidx == parent.mceqidx) and 
                parent.can_interact and not config['first_interaction_mode']):
                # Substract unity from the main diagonals
                info(10, 'substracting main C diagonal from',child.name, parent.name)
                self.C_blocks[idx][np.diag_indices(self.dim)] -= 1

            if idx not in self.C_blocks:
                continue
            self.max_lint = np.max([self.max_lint, np.max(parent.inverse_interaction_length())])
            self.C_blocks[idx] *= parent.inverse_interaction_length()
        
        self.int_m = self._csr_from_blocks(self.C_blocks)

        # -I + D
        if not skip_decay_matrix or self.dec_m is None:
            self.max_ldec = 0.
            for parent, child in product(cparts,cparts):
                idx = (child.mceqidx, parent.mceqidx)
                # Main diagonal
                if child.mceqidx == parent.mceqidx and not parent.is_stable:
                    # Substract unity from the main diagonals
                    info(10, 'substracting main D diagonal from',child.name, parent.name)
                    self.D_blocks[idx][np.diag_indices(self.dim)] -= 1.
                if idx not in self.D_blocks:
                    continue
                # Multiply with Lambda_dec
                # Track the maximal decay length for the calculation of integration steps
                self.max_ldec = max([self.max_ldec, np.max(parent.inverse_decay_length())])
                self.D_blocks[idx] *= parent.inverse_decay_length()

        self.dec_m = self._csr_from_blocks(self.D_blocks)

        for mname, mat in [('C', self.int_m), ('D', self.dec_m)]:
            mat_density = (float(mat.nnz) / float(np.prod(mat.shape)))
            info(5, "{0} Matrix info:".format(mname))
            info(5, "    density    : {0:3.2%}".format(mat_density))
            info(5, "    shape      : {0} x {1}".format(*mat.shape))
            if config['use_sparse']:
                info(5, "    nnz        : {0}".format(mat.nnz))
            info(10, "    sum        :", mat.sum())

        info(2, "Done filling matrices.")

        return self.int_m, self.dec_m
    
    def _zero_mat(self):
        """Returns a new square zero valued matrix with dimensions of grid.
        """
        return np.zeros((self.pman.dim, self.pman.dim))

    def _csr_from_blocks(self, blocks):
        """Construct a csr matrix from a dictionary of submatrices (blocks)
        
        Note::

            It's super pain the a** to construct a properly indexed sparse matrix
            directly from the blocks, since it totally messes up the order.
        """
        from scipy.sparse import csr_matrix

        new_mat = np.zeros((self.dim_states,self.dim_states))
        for (c,p), d in blocks.iteritems():
            rc, rp = self.pman.mceqidx2pref[c], self.pman.mceqidx2pref[p]
            new_mat[rc.lidx:rc.uidx,rp.lidx:rp.uidx] = d
        
        return csr_matrix(new_mat)

    def _follow_chains(self, p, pprod_mat, p_orig, idcs, propmat, reclev=0):
        """Some recursive magic.
        """
        info(10, reclev * '\t', 'entering with', p.name)

        for d in p.children:
            info(10, reclev * '\t', 'following to', d.name)
            if not d.is_resonance:
                dprop = self._zero_mat()
                self.decays.assign_d_idx(p.pdg_id, idcs, d.pdg_id, d.hadridx,
                                         dprop)
                # TODO: fix aliases
                # alias = self._alias(p, d)

                # Check if combination of mother and daughter has a special alias
                # assigned and the index has not be replaced (i.e. pi, K, prompt)
                # if not alias:
                
                propmat[(d.mceqidx,p_orig.mceqidx)] += dprop.dot(pprod_mat)
                # else:
                #     propmat[alias[0]:alias[1], pman[p_orig].lidx:pman[p_orig].
                #             uidx] += dprop.dot(pprod_mat)

                # TODO: fix aliases
                # alt_score = self._alternate_score(p, d)
                # if alt_score:
                #     propmat[alt_score[0]:alt_score[1], pman[p_orig].
                #             lidx:pman[p_orig].uidx] += dprop.dot(pprod_mat)

            if config["debug_level"] >= 10:
                pstr = 'res'
                dstr = 'Mchain'
                if idcs == p.hadridx:
                    pstr = 'prop'
                    dstr = 'Mprop'
                info(10, reclev * '\t',
                     'setting {0}[({1},{3})->({2},{4})]'.format(
                         dstr, p_orig.name, d.name, pstr, 'prop'))

            if d.is_mixed or d.is_resonance:
                dres = self._zero_mat()
                self.decays.assign_d_idx(p.pdg_id, idcs, d.pdg_id, d.residx,
                                         dres)
                reclev += 1
                self._follow_chains(d, dres.dot(pprod_mat), p_orig, d.residx,
                                    propmat, reclev)
            else:
                info(10, reclev * '\t', '\t terminating at', d.name)
    
    def _fill_matrices(self, skip_decay_matrix=False):
        """Generates the C and D matrix from scratch.
        """
        from collections import defaultdict

        if not skip_decay_matrix or self.dec_m is None:
            # Initialize empty D matrix
            self.D_blocks = defaultdict(lambda : self._zero_mat())
            for p in self.pman.cascade_particles:
                # Fill parts of the D matrix related to p as mother
                if not p.is_stable and bool(p.children):
                    self._follow_chains(
                        p,
                        np.diag(np.ones((self.dim))),
                        p,
                        p.hadridx,
                        self.D_blocks,
                        reclev=0)

        # Initialize empty C blocks
        self.C_blocks = defaultdict(lambda : self._zero_mat())
        for p in self.pman.cascade_particles:
            # if p doesn't interact, skip interaction matrices
            if (not p.is_projectile or
                (config["adv_set"]["allowed_projectiles"] and abs(
                    p.pdg_id) not in config["adv_set"]["allowed_projectiles"])):
                info(
                    2,
                    'Particle production by {0} explicitly disabled'.format(
                        p.pdg_id),
                    condition=p.is_projectile)
                continue
            elif config['adv_set']['disable_sec_interactions'] and p.pdg_id not in [
                    2212, 2112
            ]:
                info(1, 'Veto secodary interaction of', p.pdg_id)
                continue

            # go through all secondaries
            # @debug: y_matrix is copied twice in non_res and res
            # calls to assign_y_...

            for s in p.hadr_secondaries:

                if s not in self.pman.cascade_particles:
                    continue
                if 'DPMJET' in self.y.iam and s.is_lepton:
                    info(1, 'DPMJET hotfix direct leptons', s)
                    continue
                if config['adv_set']['disable_direct_leptons'] and s.is_lepton:
                    info(2, 'veto direct lepton', s)
                    continue
                if not s.is_resonance:
                    cmat = self._zero_mat()
                    self.y.assign_yield_idx(p.pdg_id, p.hadridx, s.pdg_id,
                                            s.hadridx, cmat)
                    self.C_blocks[(s.mceqidx, p.mceqidx)] += cmat
                cmat = self._zero_mat()
                self.y.assign_yield_idx(p.pdg_id, p.hadridx, s.pdg_id,
                                        s.residx, cmat)
                self._follow_chains(
                    s,
                    cmat,
                    p,
                    s.residx,
                    self.C_blocks,
                    reclev=1)