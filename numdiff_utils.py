import numpy as np

def numdiff(myfunc, x, par, nargout=1):
    '''
    computes gradient and hessian of myfunc numerically

    :param myfunc: pointer to either of f1 or f2
    :param x: Input vector R^(mx1)
    :param par: a dictionary including keys:
        'epsilon' : The incerement of x
        'f_par' : parameters dictionary given to function
        'gradient' : gradient function of f
    :param nargout: Like nargout of matlab, can be 1 or 2
    :return: [gnum, Hnum]
        gnum : Numerical estimation of function gradient
        Hnum : Numerical estimation of function Hessian
    '''
    assert callable(myfunc)
    assert isinstance(x, np.ndarray)
    assert isinstance(par, dict)
    assert 'epsilon' in par.keys()
    assert isinstance(nargout, int)
    assert nargout in range(1, 3)

    epsilon_tot = par['epsilon']
    assert isinstance(epsilon_tot, float)
    max_abs_val_of_x = max(x.min(), x.max(), key=abs)
    if max_abs_val_of_x != 0:
        epsilon = pow(epsilon_tot, 1 / 3) * max_abs_val_of_x
    else:
        epsilon = epsilon_tot**2


    # standard_base = np.array(((1, 0, 0),
    #                           (0, 1, 0),
    #                           (0, 0, 1)))

    standard_base = np.identity(len(x))

    grad = []
    for i in range(0, len(x)):
        right_g_i = myfunc(x+epsilon*standard_base[i])
        left_g_i = myfunc(x-epsilon*standard_base[i])
        g_i = (right_g_i - left_g_i)/(2*epsilon)
        grad.append(g_i)
    grad = np.array(grad)

    if nargout == 1:
        return grad

    hess = []
    analytic_grad = par['gradient']
    assert callable(analytic_grad)
    for i in range(0, len(x)):
        right_sample = analytic_grad(x+epsilon*standard_base[i], par['f_par'])
        left_sample = analytic_grad(x-epsilon*standard_base[i], par['f_par'])
        h_i = (right_sample-left_sample)/(2*epsilon)
        hess.append(h_i)
    hess = np.array(hess)

    return grad, hess