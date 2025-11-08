import pandas as pd
import numpy as np
from scipy.stats import genextreme as gev, kstest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/GMStock_2015.csv", parse_dates=["date"])
df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
df.set_index("date", inplace=True)


shift = abs(df["RET"].min()) + 1 
df["RET_log"] = np.log(df["RET"] + shift)


# Resample monthly and take maxima
block_maxima = df["RET"].resample("ME").max().dropna()

# Fit GEV
shape, loc, scale = gev.fit(block_maxima)
print(f"[Block Maxima - GEV] shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")

# Plot histogram and fitted GEV
x = np.linspace(block_maxima.min(), block_maxima.max(), 100)
pdf = gev.pdf(x, shape, loc=loc, scale=scale)

plt.figure(figsize=(10,5))
plt.hist(block_maxima, bins=50, density=True, alpha=0.6, color='skyblue', label="Monthly Max Returns")
plt.plot(x, pdf, 'r-', lw=2, label="GEV Fit")
plt.title("General Motors GEV Fit to Block Maxima of Monthly Returns")
plt.xlabel("Monthly Maximum Return")
plt.ylabel("Density")
plt.legend()
plt.show()

# --- Display MLE Table ---
fit_results = [["Block Maxima", f"{shape:.4f}", f"{loc:.4f}", f"{scale:.4f}"]]

fig, ax = plt.subplots(figsize=(6, 1.5))
ax.axis('tight')
ax.axis('off')

column_labels = ["Dataset", "Shape (k)", "Location (μ)", "Scale (σ)"]
table = ax.table(cellText=fit_results,
                 colLabels=column_labels,
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title("Maximum Likelihood Estimates for GEV Fit (Block Maxima)", pad=10)
plt.show()

# --- KS-test ---
D, p_value = kstest(block_maxima, 'genextreme', args=(shape, loc, scale))
print(f"KS-test D-statistic: {D:.4f}, p-value: {p_value:.4f}")

# --- Bootstrap for confidence intervals ---
n_bootstrap = 1000
boot_shapes = []
boot_locs = []
boot_scales = []

for _ in range(n_bootstrap):
    sample = np.random.choice(block_maxima, size=len(block_maxima), replace=True)
    s_b, loc_b, sc_b = gev.fit(sample)
    boot_shapes.append(s_b)
    boot_locs.append(loc_b)
    boot_scales.append(sc_b)

shape_ci = np.percentile(boot_shapes, [2.5, 97.5])
loc_ci = np.percentile(boot_locs, [2.5, 97.5])
scale_ci = np.percentile(boot_scales, [2.5, 97.5])

print(f"95% CI for shape: {shape_ci}")
print(f"95% CI for location: {loc_ci}")
print(f"95% CI for scale: {scale_ci}")

# --- Plot histogram, fitted PDF, and confidence intervals ---
x = np.linspace(block_maxima.min(), block_maxima.max(), 1000)
pdf = gev.pdf(x, shape, loc=loc, scale=scale)

boot_pdfs = np.array([gev.pdf(x, s, loc=l, scale=sc) 
                      for s, l, sc in zip(boot_shapes, boot_locs, boot_scales)])
ci_lower = np.percentile(boot_pdfs, 2.5, axis=0)
ci_upper = np.percentile(boot_pdfs, 97.5, axis=0)

plt.figure(figsize=(10,6))
plt.hist(block_maxima, bins=50, density=True, alpha=0.5, color='skyblue', label='Monthly Max Returns')
plt.plot(x, pdf, 'r-', lw=2, label='GEV Fit')
plt.fill_between(x, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
plt.text(0.05, max(pdf)*0.9, f'KS-test p-value: {p_value:.4f}', fontsize=12)
plt.xlabel('Monthly Maximum Return')
plt.ylabel('Density')
plt.title('GEV Fit with 95% CI and KS p-value (Block Maxima)')
plt.legend()
plt.tight_layout()
plt.show()
