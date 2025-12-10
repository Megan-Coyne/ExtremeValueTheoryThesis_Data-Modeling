import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev

def fit_gev_block_maxima(data, company, block_freq="ME", shift_log=True):
    """
    Fits a Generalized Extreme Value (GEV) distribution to block maxima of a dataset.

    Parameters
    ----------
    data : array-like
        1D numeric array of values (e.g., stock returns).
    company : str
        Company name for labeling plots/results.
    block_freq : str
        Pandas offset alias for block maxima (e.g., "ME" = monthly, "A" = annual).
    shift_log : bool
        If True, shift values so minimum > 0 before log-transform.

    Returns
    -------
    dict
        Contains:
        - 'shape', 'loc', 'scale': fitted GEV parameters
        - 'gev_dist': frozen GEV distribution object
        - 'block_maxima': array of block maxima
        - 'company': company name
    """
    import pandas as pd

    series = pd.Series(data)
    
    # Optional log transform and shift
    if shift_log:
        shift = abs(series.min()) + 1
        series = np.log(series + shift)

    # Compute block maxima
    block_maxima = series.resample(block_freq).max().dropna()

    # Fit GEV
    shape, loc, scale = gev.fit(block_maxima)
    gev_dist = gev(c=shape, loc=loc, scale=scale)

    return {
        "shape": shape,
        "loc": loc,
        "scale": scale,
        "gev_dist": gev_dist,
        "block_maxima": block_maxima.values,
        "company": company
    }


def plot_gev_fit(fit_result):
    """
    Plots histogram of block maxima and fitted GEV PDF.

    Parameters
    ----------
    fit_result : dict
        Output of `fit_gev_block_maxima`.
    """
    block_maxima = fit_result["block_maxima"]
    gev_dist = fit_result["gev_dist"]
    company = fit_result["company"]

    x_vals = np.linspace(block_maxima.min(), block_maxima.max(), 400)
    pdf_vals = gev_dist.pdf(x_vals)

    plt.figure(figsize=(8,5))
    plt.hist(block_maxima, bins=30, density=True, alpha=0.5, label="Empirical Block Maxima")
    plt.plot(x_vals, pdf_vals, linewidth=2, label="Fitted GEV PDF")
    plt.title(f"{company} — GEV Fit to Block Maxima")
    plt.xlabel("Block Maximum")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
