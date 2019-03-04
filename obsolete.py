# This is from InteractionYields
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

# From get_y_matrix
# if not self.band:
        #     return m
        # else:
        #     # set all elements except those inside selected xf band to 0
        #     m = np.copy(m)
        #     m[np.tril_indices(self.dim, self.dim - self.band[1] - 1)] = 0.
        #     # if self.band[0] < 0:
        #     m[np.triu_indices(self.dim, self.dim - self.band[0])] = 0.
        #     return m
    # def set_xf_band(self, xl_low_idx, xl_up_idx):
    #     """Limits interactions to certain range in :math:`x_{\\rm lab}`.

    #     Limit particle production to a range in :math:`x_{\\rm lab}` given
    #     by lower index, below which no particles are produced and an upper
    #     index, respectively. (Needs more clarification).

    #     Args:
    #       xl_low_idx (int): lower index of :math:`x_{\\rm lab}` value
    #       xl_up_idx (int): upper index of :math:`x_{\\rm lab}` value
    #     """

    #     if xl_low_idx >= 0 and xl_up_idx > 0:
    #         self.band = (xl_low_idx, xl_up_idx)
    #     else:
    #         self.band = None
    #         info(2, 'reset selection of x_lab band')
    #         return
    #     bins = self.energy_grid.b
    #     info(2, 'limiting Feynman x range to: {0:5.2e} - {1:5.2e}'.format(
    #                       (bins / bins[-1])[self.band[0]],
    #                       (bins / bins[-1])[self.band[1]]))

    # def __repr__(self):
    #     a_string = 'Possible (projectile,secondary) configurations:\n'
    #     for p in sorted(self.relations):
    #         if key not in ['evec', 'ebins']:
    #             a_string += str(key) + '\n'
    #     return a_string

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