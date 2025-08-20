import pandas as pd
import numpy as np
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt


df = pd.read_csv("data/FordStock.csv", parse_dates=["date"])
df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
df.set_index("date", inplace=True)


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
plt.title("Ford GEV Fit to Block Maxima of Monthly Returns")
plt.xlabel("Monthly Maximum Return")
plt.ylabel("Density")
plt.legend()
plt.show()
