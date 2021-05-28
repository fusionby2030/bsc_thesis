import numpy as np
import GPy
import sys
import scipy as scp
from tqdm import tqdm


def KLrel(X, model, delta, latent=False, likelihood='gaussian', pointwise=False):
    """Computes relevance estimates for each covariate using the KL method based on the data matrix X and a GPy model.
    The parameter delta defines the amount of perturbation used."""
    n = X.shape[0]
    p = X.shape[1]

    jitter = 1e-15

    # perturbation
    deltax = np.linspace(-delta, delta, 3)

    # loop through the data points X
    relevances = np.zeros((n, p))

    for j in range(0, n):

        x_n = np.reshape(np.repeat(X[j, :], 3), (p, 3))

        # loop through covariates
        for dim in range(0, p):

            # perturb x_n
            x_n[dim, :] = x_n[dim, :] + deltax

            # compute the KL relevance estimate
            if (latent == True):
                preddeltamean, preddeltavar = GPy.models.GPRegression.predict_noiseless(model, x_n.T, full_cov=False)
                mean_orig = np.asmatrix(np.repeat(preddeltamean[1], 3)).T
                var_orig = np.asmatrix(np.repeat(preddeltavar[1], 3)).T
                # compute the relevance estimate at x_n
                KLsqrt = np.sqrt(0.5 * (var_orig / preddeltavar + np.multiply((preddeltamean.reshape(3, 1) - mean_orig),
                                                                              (preddeltamean.reshape(3,
                                                                                                     1) - mean_orig)) / preddeltavar - 1) + np.log(
                    np.sqrt(preddeltavar / var_orig)) + jitter)
                relevances[j, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta
            elif (latent == False):
                if (likelihood == 'gaussian'):
                    preddeltamean, preddeltavar = GPy.models.GPRegression.predict(model, x_n.T, full_cov=False)
                    mean_orig = np.asmatrix(np.repeat(preddeltamean[1], 3)).T
                    var_orig = np.asmatrix(np.repeat(preddeltavar[1], 3)).T
                    # compute the relevance estimate at x_n
                    KLsqrt = np.sqrt(0.5 * (
                            var_orig / preddeltavar + np.multiply((preddeltamean.reshape(3, 1) - mean_orig), (
                            preddeltamean.reshape(3, 1) - mean_orig)) / preddeltavar - 1) + np.log(
                        np.sqrt(preddeltavar / var_orig)) + jitter)
                    relevances[j, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta
                elif (likelihood == 'binomial'):
                    preddeltamean = GPy.models.GPRegression.predict(model, x_n.T, full_cov=False)
                    mean_orig = np.array(np.asmatrix(np.repeat(preddeltamean[0][1], 3)).T)
                    # compute the relevance estimate at x_n
                    KLsqrt = np.sqrt(mean_orig * np.log(mean_orig / preddeltamean[0]) + (1.0 - mean_orig) * np.log(
                        (1.0 - mean_orig) / (1.0 - preddeltamean[0])) + jitter)
                    relevances[j, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta
                elif (likelihood == 'poisson'):
                    lambdaa = model.predict(x_n.T, full_cov=False)[0]
                    lambdaa_orig = np.array(np.asmatrix(np.repeat(lambdaa[1, 0], 3)).T)
                    KLsqrt = np.sqrt(lambdaa_orig * np.log(lambdaa_orig / lambdaa) + lambdaa - lambdaa_orig + jitter)
                    relevances[j, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta
                else:
                    sys.exit(
                        'Unknown likelihood. Either specify latent=True or change the likelihood. Possibilities: "gaussian", "binomial", "poisson".')
            else:
                sys.exit('latent must be True or False')

            # remove the perturbation
            x_n[dim, :] = x_n[dim, :] - deltax

    if (pointwise == True):
        return relevances
    else:
        return np.mean(relevances, axis=0)


def VARrel(X, model, nquadr, pointwise=False):  # now stops jittering when cond_2 < 100
    """Computes relevance estimates for each covariate using the VAR method based on the data matrix X and a GPy model.
    The parameter nquadr defines the number of quadrature points to use in Gauss-Hermite quadrature integration."""
    n = X.shape[0]
    p = X.shape[1]

    relevances = np.zeros((n, p))
    [points, weights] = np.polynomial.hermite.hermgauss(nquadr)
    jitter = 1e-9

    # full covariance matrix of X plus small jitter on diagonal
    fullcov = np.cov(X, rowvar=False) + jitter * np.eye(p)

    # if condition number is high, add a diagonal term until it goes below 100
    # this is a bit heuristic
    while (np.linalg.cond(fullcov, p=2) > 100):
        jitter = jitter * 10
        fullcov = fullcov + jitter * np.eye(p)

    # Cholesky decomposition of the covariance matrix
    cholfull = scp.linalg.cholesky(fullcov, lower=True)

    # loop through covariates
    for j in range(0, p):

        # remove j'th covariate
        jvals = X[:, j]
        nojvals = np.delete(X, (j), axis=1)
        jmean = jvals.mean()
        nojmean = nojvals.mean(axis=0)

        jcov = fullcov[j, j]
        jnojcov = fullcov[j, :]
        jnojcov = np.delete(jnojcov, (j), axis=0)
        jnojcov = jnojcov.reshape(1, p - 1)

        # Cholesky decomposition of the submatrix
        cholsub = cholsubmatrix(cholfull, j)
        meanfactor = scp.linalg.cho_solve((cholsub, True), jnojcov.T).T
        intcov = jcov - np.dot(meanfactor, jnojcov.T)

        # loop through data points
        for k in range(0, n):
            nojk = nojvals[k, :]
            intmean = jmean + np.dot(meanfactor, (nojk - nojmean))
            fcalcpoints = np.repeat(X[k, :], nquadr).reshape(p, nquadr).T
            fcalcpoints[:, j] = np.sqrt(2) * np.sqrt(intcov) * points + intmean

            predmean, predvar = GPy.models.GPRegression.predict_noiseless(model, fcalcpoints, full_cov=False)
            fsquare = predmean * predmean

            # Gauss-Hermite quadrature integration
            relevances[k, j] = np.dot(fsquare.T, weights) / np.sqrt(np.pi) - np.dot(predmean.T, weights) * np.dot(
                predmean.T, weights) / np.pi

    if (pointwise == True):
        return relevances
    else:
        return np.mean(relevances, axis=0)


# Auxiliary functions

def cholr1update(LL, xx):
    """Computes a rank-1 update to a Cholesky decomposition"""
    L = LL.copy()
    x = xx.copy()
    n = np.shape(x)[0]
    for k in np.arange(0, n):
        r = np.sqrt(L[k, k] * L[k, k] + x[k] * x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if (k < n - 1):
            L[k + 1:, k] = (L[k + 1:, k] + s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
    return L


def cholsubmatrix(LL, index):
    """Computes the Cholesky decomposition for a submatrix (matrix with one row and one column removed) using the Cholesky of the full matrix"""
    L = LL.copy()
    n = np.shape(L)[0]
    chol = np.zeros((n - 1, n - 1))
    chol[:index, :index] = L[:index, :index]
    if (index < n):
        chol[index:, :index] = L[index + 1:, :index]
        chol[index:, index:] = cholr1update(L[index + 1:, index + 1:], L[index + 1:, index])
    return chol


def kernel_relevance(x, y, kernel=None, repeats=2, m=12, delta=0.001, nquadr=11):
    # Initialize the values
    rel_ARD = np.zeros(m)
    rel_KL = np.zeros(m)
    rel_VAR = np.zeros(m)
    """
    takes a GPY kernel
    """
    repeat_iter = tqdm(range(0, repeats), desc='repeat')
    for i in repeat_iter:
        if kernel is None:
            kernel = GPy.kern.RBF(input_dim=m, ARD=True) + GPy.kern.Bias(input_dim=m)
        model = GPy.models.GPRegression(x, y, kernel, )
        model.optimize()
        repeat_iter.set_postfix(optimize=True)

        # ARD relevance value is the inverse of the length scale
        rel_ARD = rel_ARD + 1 / model.sum.rbf.lengthscale
        repeat_iter.set_postfix(ARD=True)

        # KL relevance value
        rel_KL = rel_KL + KLrel(x, model, delta)
        repeat_iter.set_postfix(KL=True)

        # VAR relevance value
        rel_VAR = rel_VAR + VARrel(x, model, nquadr)
        repeat_iter.set_postfix(VAR=True)

    rel_ARD_nor = rel_ARD / np.max(rel_ARD)
    rel_KL_nor = rel_KL / np.max(rel_KL)
    rel_VAR_nor = rel_VAR / np.max(rel_VAR)

    return rel_ARD_nor, rel_KL_nor, rel_VAR_nor