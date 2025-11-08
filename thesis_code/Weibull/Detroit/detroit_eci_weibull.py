# detroit_weibull_demo.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

# ====================================================
# 1. Load the data
# ====================================================
# Replace this with your actual file path
df = pd.read_csv("data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv")

# Example: take absolute daily returns (must be positive for Weibull)
data = df["RET"].dropna().abs()

# ====================================================
# 2. Fit Weibull distribution
# ====================================================
# Fit shape (k), location, scale (lambda)
shape, loc, scale = weibull_min.fit(data, floc=0)
print(f"Fitted Weibull parameters:\n  Shape (k) = {shape:.4f}\n  Scale (λ) = {scale:.4f}")

# ====================================================
# 3. Plot histogram vs fitted Weibull PDF
# ====================================================
x = np.linspace(0, data.max(), 200)
pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)

plt.figure(figsize=(12, 8))
plt.hist(data, bins=100, density=True, alpha=0.5, color="skyblue", label="Empirical")
plt.plot(x, pdf, "r-", lw=2, label="Fitted Weibull")
plt.xlabel("Absolute Returns")
plt.ylabel("Density")
plt.title("Weibull Fit to Stock Returns")
plt.legend()
plt.grid(True)
plt.show()

# ====================================================
# 4. Compare different years (like multiple μ in Poisson example)
# ====================================================
plt.figure(figsize=(12, 8))
for year in [2000, 2008, 2020]:
    sample = df[df["date"].astype(str).str.startswith(str(year))]["RET"].dropna().abs()
    if len(sample) == 0:
        continue
    shape_y, loc_y, scale_y = weibull_min.fit(sample, floc=0)
    x = np.linspace(0, sample.max(), 200)
    pdf_y = weibull_min.pdf(x, shape_y, loc=loc_y, scale=scale_y)
    plt.plot(x, pdf_y, lw=2, label=f"{year}: k={shape_y:.2f}, λ={scale_y:.2f}")

plt.hist(data, bins=100, density=True, alpha=0.2, label="All data")
plt.xlabel("Absolute Returns")
plt.ylabel("Density")
plt.title("Weibull Fits by Year")
plt.legend()
plt.grid(True)
plt.show()

# ====================================================
# 5. Optional: Define a Weibull PDF function (like you did for Poisson)
# ====================================================
def weibull_pdf(x, k, lam):
    """Custom Weibull PDF (location fixed at 0)."""
    return (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)

# Example: compare different parameter values
x = np.linspace(0, 2, 200)
fig, ax = plt.subplots(figsize=(12, 8))

for (k, lam) in [(1, 1), (2, 1), (5, 1), (2, 0.5)]:
    ax.plot(x, weibull_pdf(x, k, lam), lw=2, label=f"k={k}, λ={lam}")

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Weibull PDFs for Different Parameters")
ax.legend()
ax.grid(True)
plt.show()
