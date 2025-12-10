import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

def fit_gpd(data, company, threshold=None, quantile=0.95):
    """
    Fits a Generalized Pareto Distribution (GPD) to exceedances of a dataset.
    """
    data = np.asarray(data)

    if threshold is None:
        threshold = np.quantile(data, quantile)

    exceedances = data[data > threshold] - threshold
    if len(exceedances) == 0:
        raise ValueError("No data exceeds the threshold. Choose a lower threshold.")

    xi, loc, sigma = genpareto.fit(exceedances, floc=0)
    gpd_dist = genpareto(c=xi, loc=0, scale=sigma)

    return {
        "xi": xi,
        "sigma": sigma,
        "threshold": threshold,
        "gpd_dist": gpd_dist,
        "company": company
    }


def plot_gpd_fit(data, fit_result):
    """
    Plots both the empirical exceedance histogram and the fitted GPD PDF,
    along with the survival function (1-CDF) for extremes.
    """
    u = fit_result["threshold"]
    gpd = fit_result["gpd_dist"]
    company = fit_result["company"]

    exceedances = np.asarray(data)
    exceedances = exceedances[exceedances > u] - u
    exceedances_sorted = np.sort(exceedances)

    x_vals = np.linspace(0, exceedances_sorted.max(), 400)
    gpd_pdf = gpd.pdf(x_vals)
    gpd_sf = gpd.sf(x_vals) 

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(exceedances, bins=30, density=True, alpha=0.5,
             label="Empirical Exceedance Density")
    plt.plot(x_vals, gpd_pdf, linewidth=2, label="Fitted GPD PDF")
    plt.title(f"{company} — GPD PDF")
    plt.xlabel("Exceedance over threshold (x - u)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(exceedances_sorted, np.arange(1, len(exceedances_sorted)+1)[::-1]/len(exceedances_sorted), 'o', label="Empirical SF")
    plt.plot(x_vals, gpd_sf, linewidth=2, label="Fitted GPD SF")
    plt.title(f"{company} — GPD Survival Function")
    plt.xlabel("Exceedance over threshold (x - u)")
    plt.ylabel("P(X >= x)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
