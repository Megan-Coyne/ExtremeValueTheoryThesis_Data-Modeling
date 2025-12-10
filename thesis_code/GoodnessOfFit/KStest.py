import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest


def ks_test(data, dist, transform=None, n_bootstrap=500):
    """
    Kolmogorov–Smirnov test for ANY frozen SciPy distribution.

    Parameters
    ----------
    data : array-like
        Raw data sample.
    dist : scipy.stats frozen distribution
        Fitted distribution object (e.g., genpareto(c, loc, scale)).
    transform : callable or None
        Function to transform the data before testing (e.g., exceedances).
    n_bootstrap : int
        Number of bootstrap samples for PDF CI.

    Returns
    -------
    result : dict
        Contains KS statistic, p-value, and CI arrays.
    """
    data = np.asarray(data)

    if transform is not None:
        data = transform(data)
    data = data[~np.isnan(data)]

    D, p_value = kstest(data, dist.cdf)

    x = np.linspace(data.min(), data.max(), 800)
    pdf = dist.pdf(x)

    boot_pdfs = []
    base_dist_class = dist.dist 
    base_params = dist.args + tuple(dist.kwds.values())

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)

        try:
            params = base_dist_class.fit(sample)
            boot_pdf = base_dist_class(*params).pdf(x)
            boot_pdfs.append(boot_pdf)
        except Exception:
            continue

    boot_pdfs = np.array(boot_pdfs)
    ci_lower = np.percentile(boot_pdfs, 2.5, axis=0)
    ci_upper = np.percentile(boot_pdfs, 97.5, axis=0)

    return {
        "ks_statistic": D,
        "p_value": p_value,
        "x": x,
        "pdf": pdf,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "data": data
    }


def plot_ks_test(result, title="KS Test + Bootstrap CI"):
    """Plot the empirical histogram, PDF, CI region, and KS statistic."""
    data = result["data"]
    x = result["x"]

    hist_counts, hist_bins = np.histogram(data, bins=40, density=True)
    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

    plt.figure(figsize=(10, 6))

    plt.bar(
        hist_centers,
        hist_counts,
        width=hist_centers[1] - hist_centers[0],
        alpha=0.4,
        label="Empirical PDF"
    )

    plt.plot(x, result["pdf"], "r-", lw=2, label="Fitted PDF")
    plt.fill_between(x, result["ci_lower"], result["ci_upper"],
                     color="red", alpha=0.2, label="95% CI")

    plt.text(
        0.05, max(result["pdf"]) * 0.9,
        f"KS Statistic: {result['ks_statistic']:.4f}\n"
        f"p-value: {result['p_value']:.4f}",
        fontsize=12
    )

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
