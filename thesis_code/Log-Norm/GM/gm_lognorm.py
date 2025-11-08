import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Load data
file_path = "data/GMStock.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# Absolute returns
df['abs_RET'] = df['RET'].abs()
data = df['abs_RET'].dropna()

# Ensure all values > 0
data_shift = data + 1e-6 if np.any(data <= 0) else data

# Fit log-normal distribution
shape, loc, scale = lognorm.fit(data_shift, floc=0)
print(f"Log-Normal Fit Parameters:\nShape (sigma): {shape:.4f}, Scale (exp(mu)): {scale:.4f}")

# Plot histogram and PDF
plt.figure(figsize=(10,5))
plt.hist(data_shift, bins=50, density=True, alpha=0.6, color='skyblue', label='Absolute Returns')

x = np.linspace(data_shift.min(), data_shift.max(), 1000)
pdf = lognorm.pdf(x, shape, loc=0, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='Log-Normal Fit')

plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Log-Normal Fit to Absolute Returns')
plt.legend()
plt.tight_layout()
plt.show()

# Display MLE Table
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')

# MLE values
fit_results = [['Log-Normal', f'{shape:.4f}', f'{scale:.4f}', f'{loc:.4f}']]

column_labels = ["Distribution", "Shape (σ)", "Scale (exp(μ))", "Location (loc)"]
table = ax.table(cellText=fit_results,
                 colLabels=column_labels,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title("Maximum Likelihood Estimates for Log-Normal Fit", pad=20)
plt.tight_layout()
plt.show()
