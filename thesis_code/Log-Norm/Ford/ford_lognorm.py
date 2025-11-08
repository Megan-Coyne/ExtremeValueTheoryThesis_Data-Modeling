import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, kstest

# Load data
file_path = "data/FordStock_2015.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# Absolute returns
df['abs_RET'] = df['RET'].abs()
df['RET_log'] = np.log1p(df['abs_RET'])  
data = df['RET_log'].dropna()

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

# --- KS-test ---
D, p_value = kstest(data_shift, 'lognorm', args=(shape, 0, scale))
print(f"KS-test D-statistic: {D:.4f}, p-value: {p_value:.4f}")

# --- Bootstrap to get CI for KS statistic ---
n_bootstrap = 1000
ks_stats = []

for _ in range(n_bootstrap):
    sample = np.random.choice(data_shift, size=len(data_shift), replace=True)
    shape_b, loc_b, scale_b = lognorm.fit(sample, floc=0)
    D_b, _ = kstest(sample, 'lognorm', args=(shape_b, 0, scale_b))
    ks_stats.append(D_b)

ks_stats = np.array(ks_stats)
ci_lower, ci_upper = np.percentile(ks_stats, [2.5, 97.5])

print(f"95% CI for KS statistic: [{ci_lower:.4f}, {ci_upper:.4f}]")

# --- Plot histogram, PDF, and KS CI ---
x = np.linspace(data_shift.min(), data_shift.max(), 1000)
pdf = lognorm.pdf(x, shape, loc=0, scale=scale)

plt.figure(figsize=(10,6))
plt.hist(data_shift, bins=50, density=True, alpha=0.5, color='skyblue', label='Absolute Returns')
plt.plot(x, pdf, 'r-', lw=2, label='Log-Normal Fit')

# Add text box for KS statistic and CI
plt.text(0.05, max(pdf)*0.9,
         f'KS D-statistic: {D:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\np-value: {p_value:.4f}',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Log-Normal Fit with KS-Test CI')
plt.legend()
plt.tight_layout()
plt.show()
