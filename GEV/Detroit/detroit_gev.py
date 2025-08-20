import pandas as pd
import numpy as np
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv", parse_dates=["observation_date"])
df.set_index("observation_date", inplace=True)

# Resample monthly and take maxima (or yearly if preferred)
block_maxima = df["DWLAGRIDX"].resample("ME").max().dropna()

# Fit GEV
shape, loc, scale = gev.fit(block_maxima)
print(f"[Block Maxima - GEV] shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")

# Plot histogram and GEV fit
x = np.linspace(block_maxima.min(), block_maxima.max(), 200)
pdf = gev.pdf(x, shape, loc=loc, scale=scale)

plt.figure(figsize=(10,5))
plt.hist(block_maxima, bins=30, density=True, alpha=0.6, color='skyblue', label="Monthly Max DWLAGRIDX")
plt.plot(x, pdf, 'r-', lw=2, label="GEV Fit")
plt.title("GEV Fit to Block Maxima of Monthly Economic Conditions Index")
plt.xlabel("Monthly Maximum")
plt.ylabel("Density")
plt.legend()
plt.show()
