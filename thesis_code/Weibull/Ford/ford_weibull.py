import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, kstest

# Load the CSV
df = pd.read_csv("data/FordStock.csv")
df['date'] = pd.to_datetime(df['date'])


df['abs_RET'] = df['RET'].abs()
df['RET_log'] = np.log1p(df['abs_RET'])  
overall_data = df['RET_log'].dropna()

# Overall Weibull Fit
shape, loc, scale = weibull_min.fit(overall_data, floc=0)
print(f"Overall Fit - Shape (k): {shape:.4f}, Scale (λ): {scale:.4f}")

plt.figure(figsize=(10,5))
plt.hist(overall_data, bins=50, density=True, alpha=0.6, color='skyblue', label='Overall Data')
x = np.linspace(overall_data.min(), overall_data.max(), 1000)
pdf = weibull_min.pdf(x, shape, loc=0, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='Overall Weibull Fit')
plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Weibull Fit to Ford Returns - Overall')
plt.legend()
plt.show()

# Year-by-Year Weibull Fits
years = [1990, 1995, 2000, 2005, 2010]
colors = ['green','orange','purple','brown','pink']

plt.figure(figsize=(10,5))

# Prepare MLE table
fit_results = []
fit_results.append(['Overall', f'{shape:.4f}', f'{scale:.4f}'])

for year, color in zip(years, colors):
    data_year = df[df['date'].dt.year == year]['abs_RET'].dropna()
    if len(data_year) == 0:
        continue
    
    shape_y, loc_y, scale_y = weibull_min.fit(data_year, floc=0)
    print(f"Year {year}: Shape = {shape_y:.4f}, Scale = {scale_y:.4f}")
    
    # Weibull PDF only
    x_y = np.linspace(data_year.min(), data_year.max(), 1000)
    pdf_y = weibull_min.pdf(x_y, shape_y, loc=0, scale=scale_y)
    plt.plot(x_y, pdf_y, lw=2, color=color, label=f'{year} Weibull Fit')
    
    # Add to table
    fit_results.append([str(year), f'{shape_y:.4f}', f'{scale_y:.4f}'])

plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Weibull Fit to Ford Returns - Year-by-Year (PDF only)')
plt.legend()
plt.show()

# Display MLE Table as Figure
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')

column_labels = ["Dataset/Year", "Shape (b)", "Scale (a)"]
table = ax.table(cellText=fit_results, colLabels=column_labels, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title("Maximum Likelihood Estimates for Weibull Fits", pad=20)
plt.show()


# --- KS-Test for Overall Weibull ---
D_overall, p_overall = kstest(overall_data, 'weibull_min', args=(shape, 0, scale))
print(f"Overall Weibull KS-test D-statistic: {D_overall:.4f}, p-value: {p_overall:.4f}")

# --- KS-Test for Each Year ---
ks_results = []

for year in years:
    data_year = df[df['date'].dt.year == year]['abs_RET'].dropna()
    if len(data_year) == 0:
        continue
    
    shape_y, loc_y, scale_y = weibull_min.fit(data_year, floc=0)
    D_y, p_y = kstest(data_year, 'weibull_min', args=(shape_y, 0, scale_y))
    ks_results.append([year, D_y, p_y])
    print(f"Year {year} KS-test: D = {D_y:.4f}, p-value = {p_y:.4f}")

# --- Optional: plot overall histogram with KS statistic ---
x = np.linspace(overall_data.min(), overall_data.max(), 1000)
pdf = weibull_min.pdf(x, shape, loc=0, scale=scale)

plt.figure(figsize=(10,6))
plt.hist(overall_data, bins=50, density=True, alpha=0.5, color='skyblue', label='Overall Data')
plt.plot(x, pdf, 'r-', lw=2, label='Overall Weibull Fit')
plt.text(0.05, max(pdf)*0.9, 
         f'KS D: {D_overall:.4f}\np-value: {p_overall:.4f}', 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('Absolute Returns')
plt.ylabel('Density')
plt.title('Overall Weibull Fit with KS-test')
plt.legend()
plt.show()