# -*- coding: utf-8 -*-
r"""
:mod:`MCEq.time_solvers` --- ODE solvers for the forward-euler time integrator
==============================================================================

The module contains functions which are called by the forward-euler
integration routine :func:`MCEq.core.MCEqRun.forward_euler`.

The integration is part of these functions. The single step

.. math::

  \Phi_{i + 1} = \left[\boldsymbol{M}_{int} + \frac{1}{\rho(X_i)}\boldsymbol{M}_{dec}\right]
  \cdot \Phi_i \cdot \Delta X_i

with

.. math::
  \boldsymbol{M}_{int} = (-\boldsymbol{1} + \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}
  :label: int_matrix

and

.. math::
  \boldsymbol{M}_{dec} = (-\boldsymbol{1} + \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}.
  :label: dec_matrix

The functions use different libraries for sparse and dense linear algebra (BLAS):

- The default for dense or sparse matrix representations is the function :func:`solv_numpy`.
  It uses the dot-product implementation of :mod:`numpy`. Depending on the details, your
  :mod:`numpy` installation can be already linked to some BLAS library like as ATLAS or MKL,
  what typically accelerates the calculation significantly.
- The fastest version, :func:`solv_MKL_sparse`, directly interfaces to the sparse BLAS routines
  from `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ via :mod:`ctypes`. If you have the
  MKL runtime installed, this function is recommended for most purposes.
- The GPU accelerated versions :func:`solv_CUDA_dense` and :func:`solv_CUDA_sparse` are implemented
  using the cuBLAS or cuSPARSE libraries, respectively. They should be considered as experimental or
  implementation examples if you need extremely high performance. To keep Python as the main
  programming language, these interfaces are accessed via the module :mod:`numbapro`, which is part
  of the `Anaconda Accelerate <https://store.continuum.io/cshop/accelerate/>`_ package. It is free
  for academic use.

"""
import numpy as np
from mceq_config import config, dbg
from MCEq.misc import info

def solv_numpy(nsteps,
               dX,
               rho_inv,
               int_m,
               dec_m,
               phi,
               grid_idcs,
               mu_loss_handler,
               fa_vars=None):
    """:mod;`numpy` implementation of forward-euler integration.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      mu_loss_handler (object): object of type :class:`SemiLagrangianEnergyLosses`
      fa_vars (dict,optional): contains variables for first interaction mode
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    grid_sol = []
    grid_step = 0

    imc = int_m
    dmc = dec_m
    dxc = dX
    ric = rho_inv
    phc = phi

    enmuloss = config['enable_muon_energy_loss']
    muloss_min_step = config['muon_energy_loss_min_step']
    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.

    if config['FP_precision'] == 32:
        imc = int_m.astype(np.float32)
        dmc = dec_m.astype(np.float32)
        dxc = dX.astype(np.float32)
        ric = rho_inv.astype(np.float32)
        phc = phi.astype(np.float32)

    from time import time
    start = time()
    stepper = None

    # Implmentation of first interaction mode
    if config['first_interaction_mode']:

        def stepper(step):
            if step <= fa_vars['max_step']:
                return (-fa_vars['Lambda_int'] * phc + imc.dot(
                    fa_vars['fi_switch'][step] * phc) + dmc.dot(
                        ric[step] * phc)) * dxc[step]
            else:
                # Equivalent of setting interaction matrix to 0
                return (
                    -fa_vars['Lambda_int'] * phc + dmc.dot(ric[step] * phc)
                ) * dxc[step]
    else:

        def stepper(step):
            return (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]

    for step in xrange(nsteps):
        phc += stepper(step)

        dXaccum += dxc[step]

        if enmuloss and (dXaccum > muloss_min_step or step == nsteps - 1):
            mu_loss_handler.solve_step(phc, dXaccum)
            dXaccum = 0.

        if (grid_idcs and grid_step < len(grid_idcs) and
                grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(phc))
            grid_step += 1

    info(2, "Performance: {0:6.2f}ms/iteration".format(
            1e3 * (time() - start) / float(nsteps)))

    return phc, grid_sol


class CUDASparseContext(object):
    def __init__(self, int_m, dec_m, device_id=0):

        if config['FP_precision'] == 32:
            self.fl_pr = np.float32
        elif config['FP_precision'] == 64:
            self.fl_pr = np.float64
        else:
            raise Exception(
                "CUDASparseContext(): Unknown precision specified.")
        #======================================================================
        # Setup GPU stuff and upload data to it
        #======================================================================
        try:
            from accelerate.cuda.blas import Blas
            import accelerate.cuda.sparse as cusparse
            from accelerate.cuda import cuda
        except ImportError:
            raise Exception("solv_CUDA_sparse(): Numbapro CUDA libaries not " +
                            "installed.\nCan not use GPU.")

        cuda.select_device(0)
        self.cuda = cuda
        self.cusp = cusparse.Sparse()
        self.cubl = Blas()
        self.set_matrices(int_m, dec_m)

    def set_matrices(self, int_m, dec_m):
        import accelerate.cuda.sparse as cusparse
        from accelerate.cuda import cuda

        self.m, self.n = int_m.shape
        self.int_m_nnz = int_m.nnz
        self.int_m_csrValA = cuda.to_device(int_m.data.astype(self.fl_pr))
        self.int_m_csrRowPtrA = cuda.to_device(int_m.indptr)
        self.int_m_csrColIndA = cuda.to_device(int_m.indices)

        self.dec_m_nnz = dec_m.nnz
        self.dec_m_csrValA = cuda.to_device(dec_m.data.astype(self.fl_pr))
        self.dec_m_csrRowPtrA = cuda.to_device(dec_m.indptr)
        self.dec_m_csrColIndA = cuda.to_device(dec_m.indices)

        self.descr = self.cusp.matdescr()
        self.descr.indexbase = cusparse.CUSPARSE_INDEX_BASE_ZERO
        self.cu_delta_phi = self.cuda.device_array_like(
            np.zeros(self.m, dtype=self.fl_pr))

    def set_phi(self, phi):
        self.cu_curr_phi = self.cuda.to_device(phi.astype(self.fl_pr))

    def get_phi(self):
        return self.cu_curr_phi.copy_to_host()

    def do_step(self, rho_inv, dX):

        self.cusp.csrmv(
            trans='N',
            m=self.m,
            n=self.n,
            nnz=self.int_m_nnz,
            descr=self.descr,
            alpha=self.fl_pr(1.0),
            csrVal=self.int_m_csrValA,
            csrRowPtr=self.int_m_csrRowPtrA,
            csrColInd=self.int_m_csrColIndA,
            x=self.cu_curr_phi,
            beta=self.fl_pr(0.0),
            y=self.cu_delta_phi)
        # print np.sum(cu_curr_phi.copy_to_host())
        self.cusp.csrmv(
            trans='N',
            m=self.m,
            n=self.n,
            nnz=self.dec_m_nnz,
            descr=self.descr,
            alpha=self.fl_pr(rho_inv),
            csrVal=self.dec_m_csrValA,
            csrRowPtr=self.dec_m_csrRowPtrA,
            csrColInd=self.dec_m_csrColIndA,
            x=self.cu_curr_phi,
            beta=self.fl_pr(1.0),
            y=self.cu_delta_phi)
        self.cubl.axpy(
            alpha=self.fl_pr(dX), x=self.cu_delta_phi, y=self.cu_curr_phi)


def solv_CUDA_sparse(nsteps,
                     dX,
                     rho_inv,
                     context,
                     phi,
                     grid_idcs,
                     mu_loss_handler):
    """`NVIDIA CUDA cuSPARSE <https://developer.nvidia.com/cusparse>`_ implementation
    of forward-euler integration.

    Function requires a working :mod:`accelerate` installation.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      mu_loss_handler (object): object of type :class:`SemiLagrangianEnergyLosses`
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    c = context
    c.set_phi(phi)

    enmuloss = config['enable_muon_energy_loss']
    muloss_min_step = config['muon_energy_loss_min_step']

    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.

    grid_step = 0
    grid_sol = []

    from time import time
    start = time()

    for step in xrange(nsteps):
        c.solve_step(rho_inv[step], dX[step])

        dXaccum += dX[step]

        if enmuloss and (dXaccum > muloss_min_step or step == nsteps - 1):
            # Download current solution vector to host
            phc = c.get_phi()
            mu_loss_handler.solve_step(phc, dXaccum)
            # Upload changed vector back..
            c.set_phi(phc)
            dXaccum = 0.

        if (grid_idcs and grid_step < len(grid_idcs) and
                grid_idcs[grid_step] == step):
            grid_sol.append(c.get_phi())
            grid_step += 1

    info(2, "Performance: {0:6.2f}ms/iteration".format(
            1e3 * (time() - start) / float(nsteps)))

    return c.get_phi(), grid_sol


def solv_MKL_sparse(nsteps,
                    dX,
                    rho_inv,
                    int_m,
                    dec_m,
                    phi,
                    grid_idcs,
                    mu_loss_handler):
    """`Intel MKL sparse BLAS
    <https://software.intel.com/en-us/articles/intel-mkl-sparse-blas-overview?language=en>`_
    implementation of forward-euler integration.

    Function requires that the path to the MKL runtime library ``libmkl_rt.[so/dylib]``
    defined in the config file.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (list): indices at which longitudinal solutions have to be saved.
      mu_loss_handler (object): object of type :class:`SemiLagrangianEnergyLosses`
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    from ctypes import cdll, c_int, c_char, POINTER, byref

    try:
        mkl = cdll.LoadLibrary(config['MKL_path'])
    except OSError:
        raise Exception("solv_MKL_sparse(): MKL runtime library not " +
                        "found. Please check path.")

    gemv = None
    axpy = None
    np_fl = None
    if config['FP_precision'] == 32:
        from ctypes import c_float as fl_pr
        # sparse CSR-matrix x dense vector
        gemv = mkl.mkl_scsrmv
        # dense vector + dense vector
        axpy = mkl.cblas_saxpy
        np_fl = np.float32
    elif config['FP_precision'] == 64:
        from ctypes import c_double as fl_pr
        # sparse CSR-matrix x dense vector
        gemv = mkl.mkl_dcsrmv
        # dense vector + dense vector
        axpy = mkl.cblas_daxpy
        np_fl = np.float64
    else:
        raise Exception("solv_MKL_sparse(): Unknown precision specified.")

    # Set number of threads
    mkl.mkl_set_num_threads(byref(c_int(config['MKL_threads'])))

    # Prepare CTYPES pointers for MKL sparse CSR BLAS
    int_m_data = int_m.data.ctypes.data_as(POINTER(fl_pr))
    int_m_ci = int_m.indices.ctypes.data_as(POINTER(c_int))
    int_m_pb = int_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    int_m_pe = int_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    dec_m_data = dec_m.data.ctypes.data_as(POINTER(fl_pr))
    dec_m_ci = dec_m.indices.ctypes.data_as(POINTER(c_int))
    dec_m_pb = dec_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    dec_m_pe = dec_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    npphi = np.copy(phi).astype(np_fl)
    phi = npphi.ctypes.data_as(POINTER(fl_pr))
    npdelta_phi = np.zeros_like(npphi)
    delta_phi = npdelta_phi.ctypes.data_as(POINTER(fl_pr))

    trans = c_char('n')
    npmatd = np.chararray(6)
    npmatd[0] = 'G'
    npmatd[3] = 'C'
    matdsc = npmatd.ctypes.data_as(POINTER(c_char))
    m = c_int(int_m.shape[0])
    cdzero = fl_pr(0.)
    cdone = fl_pr(1.)
    cione = c_int(1)

    enmuloss = config['enable_muon_energy_loss']
    muloss_min_step = config['muon_energy_loss_min_step']
    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.

    grid_step = 0
    grid_sol = []

    from time import time
    start = time()

    for step in xrange(nsteps):
        # delta_phi = int_m.dot(phi)
        gemv(
            byref(trans),
            byref(m),
            byref(m),
            byref(cdone), matdsc, int_m_data, int_m_ci, int_m_pb, int_m_pe,
            phi, byref(cdzero), delta_phi)
        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        gemv(
            byref(trans),
            byref(m),
            byref(m),
            byref(fl_pr(rho_inv[step])), matdsc, dec_m_data, dec_m_ci,
            dec_m_pb, dec_m_pe, phi, byref(cdone), delta_phi)
        # phi = delta_phi * dX + phi
        axpy(m, fl_pr(dX[step]), delta_phi, cione, phi, cione)

        dXaccum += dX[step]

        if enmuloss and (dXaccum > muloss_min_step or step == nsteps - 1):
            mu_loss_handler.solve_step(npphi, dXaccum)
            dXaccum = 0.

        if (grid_idcs and grid_step < len(grid_idcs) and
                grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(npphi))
            grid_step += 1

    info(2, "Performance: {0:6.2f}ms/iteration".format(
            1e3 * (time() - start) / float(nsteps)))

    return npphi, grid_sol

# TODO: Debug this and transition to BDF
    def _odepack(dXstep=.1,
                 initial_depth=0.0,
                 int_grid=None,
                 grid_var='X',
                 *args,
                 **kwargs):
        """Solves the transport equations with solvers from ODEPACK.

        Args:
          dXstep (float): external step size (adaptive sovlers make more steps internally)
          initial_depth (float): starting depth in g/cm**2
          int_grid (list): list of depths at which results are recorded
          grid_var (str): Can be depth `X` or something else (currently only `X` supported)

        """
        from scipy.integrate import ode
        ri = self.density_model.r_X2rho

        if config['enable_muon_energy_loss']:
            raise NotImplementedError(
                'Energy loss not imlemented for this solver.')

        # Functional to solve
        def dPhi_dX(X, phi, *args):
            return self.int_m.dot(phi) + self.dec_m.dot(ri(X) * phi)

        # Jacobian doesn't work with sparse matrices, and any precision
        # or speed advantage disappear if used with dense algebra
        def jac(X, phi, *args):
            # print 'jac', X, phi
            return (self.int_m + self.dec_m * ri(X)).todense()

        # Initial condition
        phi0 = np.copy(self.phi0)

        # Initialize variables
        grid_sol = []

        # Setup solver
        r = ode(dPhi_dX).set_integrator(
            with_jacobian=False, **config['ode_params'])

        if int_grid is not None:
            initial_depth = int_grid[0]
            int_grid = int_grid[1:]
            max_X = int_grid[-1]
            grid_sol.append(phi0)

        else:
            max_X = self.density_model.max_X

        info(
            1,
            'your X-grid is shorter then the material',
            condition=max_X < self.density_model.max_X)
        info(
            1,
            'your X-grid exceeds the dimentsions of the material',
            condition=max_X > self.density_model.max_X)

        # Initial value
        r.set_initial_value(phi0, initial_depth)

        info(
            2, 'initial depth: {0:3.2e}, maximal depth {1:}'.format(
                initial_depth, max_X))

        start = time()
        if int_grid is None:
            i = 0
            while r.successful() and (r.t + dXstep) < max_X - 1:
                info(5, "Solving at depth X =", r.t, condition=(i % 5000) == 0)
                r.integrate(r.t + dXstep)
                i += 1
            if r.t < max_X:
                r.integrate(max_X)
            # Do last step to make sure the rational number max_X is reached
            r.integrate(max_X)
        else:
            for i, Xi in enumerate(int_grid):
                info(5, 'integrating at X =', Xi, condition=i % 10 == 0)

                while r.successful() and (r.t + dXstep) < Xi:
                    r.integrate(r.t + dXstep)

                # Make sure the integrator arrives at requested step
                r.integrate(Xi)
                # Store the solution on grid
                grid_sol.append(r.y)

        info(2,
             'time elapsed during integration: {1} sec'.format(time() - start))

        self.solution = r.y
        self.grid_sol = grid_sol