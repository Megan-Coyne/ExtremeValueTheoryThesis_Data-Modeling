import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

def fit_weibull(data, company="Company", years=None, shift_log=True):
    """
    Fit Weibull distribution to overall data and optionally by year (PDF plots only).

    Parameters
    ----------
    data : pd.Series or np.ndarray
        1D numeric array (absolute returns or log-transformed returns)
    company : str
        Company name for labeling plots
    years : list[int] or None
        Optional list of years to do year-by-year fits
    shift_log : bool
        If True, shift values so min > 0 and take log(1+x)

    Returns
    -------
    dict
        Contains:
        - 'overall': dict with 'shape', 'loc', 'scale', 'dist'
        - 'yearly': dict keyed by year with same info as overall
    """
    data = pd.Series(data).dropna()

    # Optional shift + log
    if shift_log:
        shift = abs(data.min()) + 1
        data_transformed = np.log1p(data + shift - 1)
    else:
        data_transformed = data.values

    # --- Overall Fit ---
    shape, loc, scale = weibull_min.fit(data_transformed, floc=0)
    overall_dist = weibull_min(c=shape, loc=0, scale=scale)
    overall_fit = {"shape": shape, "loc": loc, "scale": scale, "dist": overall_dist}

    # Plot Overall Histogram + PDF
    x_vals = np.linspace(data_transformed.min(), data_transformed.max(), 1000)
    pdf_vals = overall_dist.pdf(x_vals)
    plt.figure(figsize=(10,5))
    plt.hist(data_transformed, bins=50, density=True, alpha=0.5, color='skyblue', label='Data')
    plt.plot(x_vals, pdf_vals, 'r-', lw=2, label='Overall Weibull Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{company} — Overall Weibull Fit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Yearly Fits and Plots ---
    yearly_fit = {}
    if years is not None:
        colors = plt.cm.tab10.colors
        plt.figure(figsize=(10,5))
        for i, year in enumerate(years):
            data_year = data[data.index.year == year].dropna()
            if len(data_year) == 0:
                continue
            shape_y, loc_y, scale_y = weibull_min.fit(data_year, floc=0)
            dist_y = weibull_min(c=shape_y, loc=0, scale=scale_y)
            yearly_fit[year] = {"shape": shape_y, "loc": loc_y, "scale": scale_y, "dist": dist_y}

            x_y = np.linspace(data_year.min(), data_year.max(), 1000)
            pdf_y = dist_y.pdf(x_y)
            plt.plot(x_y, pdf_y, lw=2, color=colors[i % len(colors)], label=f'{year} Weibull Fit')

        if yearly_fit:
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'{company} — Yearly Weibull Fits')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return {
        "overall": overall_fit,
        "yearly": yearly_fit
    }
