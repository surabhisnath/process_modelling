import numpy as np
from scipy.optimize import minimize
from numdifftools import Hessian

def _Hess_diag(fun, x, dx=1e-4):
    '''Evaluate the diagonal elements of the hessian matrix using the 3 point
    central difference formula with spacing of dx between points.'''
    n = len(x)
    hessdiag = np.zeros(n)
    for i in range(n):
        dx_i    = np.zeros(n)
        dx_i[i] = dx
        hessdiag[i] = (fun(x + dx_i) + fun(x - dx_i) - 2. * fun(x)) / (dx ** 2.)
    return hessdiag

def compute_log_likelihood(data_participant, pars_bounded, model_name):
    # call the correct ll func

def MAP(data_participant, pop_means, pop_vars, model_name):
    n_params = get_num_params(model_name)
    param_ranges = get_param_ranges(model_name)

    # initial guess
    x0 = pop_means

    # bounds
    bounds = param_ranges

    # negative log posterior
    def neg_log_post(pars):

        pars_bounded = trans_to_bounded(pars, param_ranges)
        log_lik = compute_log_likelihood(
            data_participant, pars_bounded, model_name)
        log_prior = - (len(pars) / 2.) * np.log(2 * np.pi) - np.sum(np.log(pop_vars)) \
            / 2. - sum((pars - pop_means) ** 2. / (2 * pop_vars))
        return -(log_lik + log_prior)

    # optimization
    res = minimize(neg_log_post, x0, method='L-BFGS-B', bounds=bounds)
    hess_func = neg_log_post
    diag_hess = _Hess_diag(hess_func, res['x'])

    fit_participant = {'par_u': res.x, 'diag_hess': diag_hess,
                       'log_post': -res.fun, 'success': res.success}

    return fit_participant


def em(data, num_participants, model_name, max_iter=100, tol=1e-6):
    n_params = get_num_params(model_name)
    param_ranges = get_param_ranges(model_name)

    # initialise prior
    pop_means = np.random.randn(n_params)  # or np.zeros(n_params)
    pop_vars = np.ones(n_params)  # * 6.25

    # EM algorithm

    for iteration in range(max_iter):

        # E-step
        fit_participants = []
        for i in range(num_participants):

            fit_participant = MAP(data[i], pop_means, pop_vars, model_name)
            fit_participants.append(fit_participant)

        # M-step
        pars_U = [fit_participant['par_u']
                  for fit_participant in fit_participants]
        diag_hess = [fit_participant['diag_hess']
                     for fit_participant in fit_participants]
        new_pop_means = np.mean(pars_U)
        new_pop_vars = np.mean(pars_U**2. + 1./diag_hess, 0)-pop_means**2.

        # check convergence
        if np.max(np.abs(new_pop_means-pop_means)) < tol and np.max(
                np.abs(new_pop_vars-pop_vars)) < tol:

            print(f'Converged in {iteration} iterations.')
            pop_means = new_pop_means
            pop_vars = new_pop_vars
            break

        pop_means = new_pop_means
        pop_vars = new_pop_vars

    bic = compute_bic(data, fit_participants, pop_means,
                      pop_vars, model_name)

    fit_pop = {'pop_means': pop_means, 'pop_vars': pop_vars, 'bic': bic,
               'fit_participants': fit_participants, 'model_name': model_name}

    return fit_pop