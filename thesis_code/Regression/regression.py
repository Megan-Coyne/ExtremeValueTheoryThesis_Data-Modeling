import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def distribution_regression(dist, x=None, n_points=100, transform="cdf", plot=True):
    """
    Perform regression on a fitted distribution's PDF or CDF.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        The fitted distribution (e.g., genpareto, genextreme, weibull_min).
    x : array-like or None
        Values at which to evaluate the distribution. If None, auto-generate 100 points.
    n_points : int
        Number of points to generate if x is None.
    transform : str
        "cdf" or "pdf" - whether to regress on the CDF or PDF values.
    plot : bool
        If True, show scatter + regression line.

    Returns
    -------
    dict
        Contains:
            - 'x': x-values used
            - 'y': target values (CDF or PDF)
            - 'model': fitted sklearn LinearRegression object
            - 'y_pred': predicted values from regression
    """
    if x is None:
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), n_points)

    if transform.lower() == "cdf":
        y = dist.cdf(x)
    elif transform.lower() == "pdf":
        y = dist.pdf(x)
    else:
        raise ValueError("transform must be 'cdf' or 'pdf'")

    # Reshape x for sklearn
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    if plot:
        plt.figure(figsize=(8,5))
        plt.scatter(x, y, label=f'{transform.upper()} values', color='blue')
        plt.plot(x, y_pred, 'r-', label='Regression line', lw=2)
        plt.xlabel('x')
        plt.ylabel(transform.upper())
        plt.title(f'Regression on distribution {transform.upper()}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {"x": x, "y": y, "model": model, "y_pred": y_pred}

def tail_regression(fit_dist, x_data, tail="upper", quantile=0.95, plot=True):
    """
    Perform regression on the log-tail of any fitted SciPy distribution.

    Parameters
    ----------
    fit_dist : scipy.stats frozen distribution
        A frozen SciPy distribution (e.g., genpareto(c, loc, scale)).
    x_data : array-like
        Original data used to fit the distribution.
    tail : str
        'upper' for right-tail, 'lower' for left-tail.
    quantile : float
        Quantile threshold for defining the tail (e.g., 0.95 for top 5%).
    plot : bool
        Whether to plot the tail and regression line.

    Returns
    -------
    result : dict
        - 'x_tail', 'y_tail' : tail values used in regression
        - 'slope', 'intercept' : linear regression coefficients
        - 'model' : fitted LinearRegression object
    """
    x_data = np.asarray(x_data)

    # Select tail
    if tail == "upper":
        threshold = np.quantile(x_data, quantile)
        x_tail = x_data[x_data > threshold]
    elif tail == "lower":
        threshold = np.quantile(x_data, 1 - quantile)
        x_tail = x_data[x_data < threshold]
    else:
        raise ValueError("tail must be 'upper' or 'lower'")

    # Log PDF for the tail
    y_tail = np.log(fit_dist.pdf(x_tail))

    # Linear regression
    model = LinearRegression()
    X = x_tail.reshape(-1, 1)
    model.fit(X, y_tail)
    slope = model.coef_[0]
    intercept = model.intercept_

    if plot:
        plt.figure(figsize=(8,5))
        plt.scatter(x_tail, y_tail, alpha=0.6, label="Tail log-PDF")
        plt.plot(x_tail, model.predict(X), 'r-', lw=2, label="Regression line")
        plt.xlabel("Data (tail)")
        plt.ylabel("log(PDF)")
        plt.title(f"{tail.capitalize()} Tail Log-PDF Regression")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "x_tail": x_tail,
        "y_tail": y_tail,
        "slope": slope,
        "intercept": intercept,
        "model": model
    }


import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import statsmodels.api as sm

def gpd_loglik_regression(params, X, exceedances):
    """
    Log-likelihood for GPD exceedances, where shape and scale depend linearly on X.

    params: vector containing [beta_xi (for each covariate), beta_sigma (for each covariate)]
    X: shape (n_samples, n_covariates)
    exceedances: the (Y - u) values > 0
    """
    n, p = X.shape
    # split params
    beta_xi = params[:p]
    beta_log_sigma = params[p:]  # model scale on log-scale to ensure positivity

    # compute conditional xi and sigma
    xi = X @ beta_xi  # shape parameter (can be positive or negative depending)
    log_sigma = X @ beta_log_sigma
    sigma = np.exp(log_sigma)

    # GPD log-likelihood
    # PDF: f(z) = (1/σ) * (1 + ξ z / σ)^(-1/ξ - 1)
    z = exceedances
    # constraint: 1 + xi * z / sigma > 0
    if np.any(1 + xi * z / sigma <= 0):
        return np.inf  # invalid, penalize

    ll = -np.sum(
        np.log(sigma)
        + (1/xi + 1) * np.log(1 + xi * z / sigma)
    )
    # return negative LL for minimizer
    return ll
