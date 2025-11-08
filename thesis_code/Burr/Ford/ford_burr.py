import pandas as pd
import numpy as np
from scipy.stats import burr12
import matplotlib.pyplot as plt

df = pd.read_csv("data/FordStock.csv")
df['date'] = pd.to_datetime(df['date'])
df['abs_RET'] = df['RET'].abs()

data = df['abs_RET'].dropna()

# Burr fit (all values > 0, loc=0)
data_shift = data + 1e-6  # tiny shift to avoid zeros
c, d, loc, scale = burr12.fit(data_shift, floc=0)

print(f"Burr Fit Parameters (MLE): c={c:.4f}, d={d:.4f}, loc={loc:.4f}, scale={scale:.4f}")

# Plot PDF
plt.figure(figsize=(10,5))
plt.hist(data_shift, bins=50, density=True, alpha=0.6, color='skyblue', label='Absolute Returns')

x = np.linspace(data_shift.min(), data_shift.max(), 1000)
pdf = burr12.pdf(x, c, d, loc=0, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='Burr Fit (MLE)')

plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Burr Fit to Absolute Returns (Ford)')
plt.legend()
plt.show()


# Display MLE Table
fit_results = [
               ['Ford', f'{c:.4f}', f'{d:.4f}', f'{loc:.4f}', f'{scale:.4f}']]

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis('tight')
ax.axis('off')

column_labels = ["Dataset", "c", "d", "loc", "Scale"]
table = ax.table(cellText=fit_results,
                 colLabels=column_labels,
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title("Maximum Likelihood Estimates for Burr Fit", pad=20)
plt.tight_layout()
plt.show()
