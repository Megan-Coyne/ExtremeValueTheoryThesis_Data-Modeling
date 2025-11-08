import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from scipy.integrate import simpson

# Load data
df = pd.read_csv("data/FordStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df['abs_RET'] = df['RET'].abs()
df['RET_log'] = np.log1p(df['abs_RET'])  
overall_data = df['abs_RET'].dropna()

# Function for POT + GPD fit
def pot_gpd_fit(data, threshold_quantile=0.95):
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data[data > threshold] - threshold  # Peaks over threshold
    shape, loc, scale = genpareto.fit(exceedances, floc=0)  # GPD fit
    return threshold, exceedances, shape, scale

# Overall POT + GPD
threshold, exceedances, shape, scale = pot_gpd_fit(overall_data, threshold_quantile=0.95)
print(f"Overall GPD Fit (POT > {threshold:.4f}) - Shape (ξ): {shape:.4f}, Scale (β): {scale:.4f}")

# Plot overall exceedances and GPD fit
plt.figure(figsize=(10,5))
plt.hist(exceedances, bins=50, density=True, alpha=0.6, color='skyblue', label='Exceedances')

x = np.linspace(exceedances.min(), exceedances.max(), 1000)
pdf = genpareto.pdf(x, shape, loc=0, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='GPD Fit (POT)')

area = simpson(pdf, x)
print(f"Area under PDF = {area:.4f}")

plt.xlabel('Exceedances over threshold')
plt.ylabel('Density')
plt.title(f'GPD Fit to Absolute Returns (Overall, Threshold={threshold:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# Year-by-Year POT + GPD
years = [1990, 1995, 2000, 2005, 2010]
colors = ['green','orange','purple','brown','pink']

plt.figure(figsize=(10,5))
fit_results = []
fit_results.append(['Overall', f'{shape:.4f}', f'{scale:.4f}', f'{threshold:.4f}'])

for year, color in zip(years, colors):
    data_year = df[df['date'].dt.year == year]['RET_log'].dropna()
    if len(data_year) == 0:
        continue

    threshold_y, exceedances_y, shape_y, scale_y = pot_gpd_fit(data_year)
    print(f"Year {year} - Threshold: {threshold_y:.4f}, Shape = {shape_y:.4f}, Scale = {scale_y:.4f}")

    x_y = np.linspace(exceedances_y.min(), exceedances_y.max(), 1000)
    pdf_y = genpareto.pdf(x_y, shape_y, loc=0, scale=scale_y)
    plt.plot(x_y, pdf_y, lw=2, color=color, label=f'{year} GPD Fit (POT> {threshold_y:.4f})')

    fit_results.append([str(year), f'{shape_y:.4f}', f'{scale_y:.4f}', f'{threshold_y:.4f}'])

plt.xlabel('Exceedances over threshold')
plt.ylabel('Density')
plt.title('GPD Fits Year-by-Year (POT)')
plt.legend()
plt.tight_layout()
plt.show()

# Display MLE Table
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')

column_labels = ["Dataset/Year", "Shape (ξ)", "Scale (β)", "Threshold (u)"]
table = ax.table(cellText=fit_results,
                 colLabels=column_labels,
                 loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title("Maximum Likelihood Estimates for GPD Fits (POT)", pad=20)
plt.tight_layout()
plt.show()

from scipy.stats import kstest

threshold, exceedances, shape, scale = pot_gpd_fit(overall_data)
print(f"Overall GPD Fit (POT > {threshold:.4f}) - Shape (ξ): {shape:.4f}, Scale (β): {scale:.4f}")

# KS-test
D, p_value = kstest(exceedances, 'genpareto', args=(shape, 0, scale))
print(f"KS-test D-statistic: {D:.4f}, p-value: {p_value:.4f}")

# Bootstrap for confidence intervals
n_bootstrap = 1000
boot_shapes = []
boot_scales = []
for _ in range(n_bootstrap):
    sample = np.random.choice(exceedances, size=len(exceedances), replace=True)
    shape_b, loc_b, scale_b = genpareto.fit(sample, floc=0)
    boot_shapes.append(shape_b)
    boot_scales.append(scale_b)

boot_shapes = np.array(boot_shapes)
boot_scales = np.array(boot_scales)

# Plot histogram, fitted PDF, and confidence intervals
x = np.linspace(exceedances.min(), exceedances.max(), 1000)
pdf = genpareto.pdf(x, shape, loc=0, scale=scale)

boot_pdfs = np.array([genpareto.pdf(x, s, loc=0, scale=sc) for s, sc in zip(boot_shapes, boot_scales)])
ci_lower = np.percentile(boot_pdfs, 2.5, axis=0)
ci_upper = np.percentile(boot_pdfs, 97.5, axis=0)

plt.figure(figsize=(10,6))
plt.hist(exceedances, bins=50, density=True, alpha=0.5, color='skyblue', label='Exceedances')
plt.plot(x, pdf, 'r-', lw=2, label='GPD Fit (POT)')
plt.fill_between(x, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
plt.text(0.05, max(pdf)*0.9, f'KS-test p-value: {p_value:.4f}', fontsize=12, color='black')
plt.xlabel('Exceedances over threshold')
plt.ylabel('Density')
plt.title(f'GPD Fit with 95% CI and KS p-value (Threshold={threshold:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# Marshall-Olkin transformation
alpha = 0.5  # can adjust depending on how heavy you want the tail
df['RET_MO'] = df['abs_RET'] / (1 - alpha * df['abs_RET'])
mo_data = df['RET_MO'].dropna()

# Function for POT + GPD fit (same as before)
threshold_mo, exceedances_mo, shape_mo, scale_mo = pot_gpd_fit(mo_data, threshold_quantile=0.95)
print(f"MO GPD Fit (POT > {threshold_mo:.4f}) - Shape (ξ): {shape_mo:.4f}, Scale (β): {scale_mo:.4f}")

# Plot histogram and GPD fit
plt.figure(figsize=(10,5))
plt.hist(exceedances_mo, bins=50, density=True, alpha=0.6, color='lightgreen', label='MO Exceedances')

x_mo = np.linspace(exceedances_mo.min(), exceedances_mo.max(), 1000)
pdf_mo = genpareto.pdf(x_mo, shape_mo, loc=0, scale=scale_mo)
plt.plot(x_mo, pdf_mo, 'r-', lw=2, label='GPD Fit (MO POT)')

plt.xlabel('Exceedances over threshold (Marshall-Olkin)')
plt.ylabel('Density')
plt.title(f'GPD Fit to MO-Transformed Returns (Threshold={threshold_mo:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# Compare original log-transformed vs Marshall-Olkin
plt.figure(figsize=(10,5))
plt.hist(overall_data, bins=50, density=True, alpha=0.5, color='skyblue', label='Log-transformed')
plt.hist(mo_data, bins=50, density=True, alpha=0.5, color='lightgreen', label='MO-transformed')
plt.xlabel('Transformed Returns')
plt.ylabel('Density')
plt.title('Comparison: Log vs Marshall-Olkin Transformation')
plt.legend()
plt.tight_layout()
plt.show()
