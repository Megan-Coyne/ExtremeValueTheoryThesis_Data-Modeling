import pandas as pd
import numpy as np
from scipy.stats import burr12, kstest
import matplotlib.pyplot as plt

df = pd.read_csv("data/StellantisStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df['abs_RET'] = df['RET'].abs()

data = df['abs_RET'].dropna()

# Burr fit (all values > 0, loc=0)
data_shift = data + 1e-6  # tiny shift to avoid zeros
c, k, loc, alpha = burr12.fit(data_shift, floc=0)

print(f"Burr Fit Parameters (MLE): c={c:.4f}, k={k:.4f}, loc={loc:.4f}, α={alpha:.4f}")

# Plot PDF
plt.figure(figsize=(10,5))
plt.hist(data_shift, bins=50, density=True, alpha=0.6, color='skyblue', label='Absolute Returns')

x = np.linspace(data_shift.min(), data_shift.max(), 1000)
pdf = burr12.pdf(x, c, k, loc=0, scale=alpha)
plt.plot(x, pdf, 'r-', lw=2, label='Burr Fit (MLE)')

plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Burr Fit to Absolute Returns (Stellantis)')
plt.legend()
plt.show()

# Display MLE Table
fit_results = [
               ['Stellantis', f'{c:.4f}', f'{k:.4f}', f'{loc:.4f}', f'{alpha:.4f}']]

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis('tight')
ax.axis('off')

column_labels = ["Dataset", "c", "k", "loc", "α"]
table = ax.table(cellText=fit_results,
                 colLabels=column_labels,
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title("Maximum Likelihood Estimates for Burr Fit", pad=20)
plt.tight_layout()
plt.show()

# --- KS-test ---
D, p_value = kstest(data_shift, 'burr12', args=(c, k, loc, alpha))
print(f"KS-test D-statistic: {D:.4f}, p-value: {p_value:.4f}")

# --- Bootstrap confidence intervals ---
n_bootstrap = 1000
boot_c, boot_k, boot_alpha = [], [], []

for _ in range(n_bootstrap):
    sample = np.random.choice(data_shift, size=len(data_shift), replace=True)
    c_b, k_b, loc_b, alpha_b = burr12.fit(sample, floc=0)
    boot_c.append(c_b)
    boot_k.append(k_b)
    boot_alpha.append(alpha_b)

boot_c = np.array(boot_c)
boot_k = np.array(boot_k)
boot_alpha = np.array(boot_alpha)

# --- PDF and confidence bands ---
x = np.linspace(data_shift.min(), data_shift.max(), 1000)
pdf = burr12.pdf(x, c, k, loc=0, scale=alpha)

boot_pdfs = np.array([burr12.pdf(x, c_b, k_b, loc=0, scale=alpha_b) 
                      for c_b, k_b, alpha_b in zip(boot_c, boot_k, boot_alpha)])
ci_lower = np.percentile(boot_pdfs, 2.5, axis=0)
ci_upper = np.percentile(boot_pdfs, 97.5, axis=0)

# --- Plot histogram, PDF, and confidence intervals ---
plt.figure(figsize=(10,6))
plt.hist(data_shift, bins=50, density=True, alpha=0.5, color='skyblue', label='Absolute Returns')
plt.plot(x, pdf, 'r-', lw=2, label='Burr Fit (MLE)')
plt.fill_between(x, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
plt.text(0.05, max(pdf)*0.9, f'KS-test p-value: {p_value:.4f}', fontsize=12)
plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Burr Fit with 95% CI and KS-test (Stellantis)')
plt.legend()
plt.tight_layout()
plt.show()
