import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from mpl_toolkits.mplot3d import Axes3D

# Files and labels
files = {
    'Ford': 'data/FordStock.csv',
    'GM': 'data/GMStock.csv',
    'Stellantis': 'data/StellantisStock.csv'
}

y_labels = list(files.keys())
y_vals = np.arange(len(y_labels))

# POT + GPD fitting for each company
exceed_dict = {}
all_exceed_max = 0

for company in y_labels:
    df = pd.read_csv(files[company])
    df['date'] = pd.to_datetime(df['date'])
    df['abs_RET'] = df['RET'].abs()
    data = df['abs_RET'].dropna()
    
    # Peaks Over Threshold
    threshold = np.quantile(data, 0.95)
    exceedances = data[data > threshold] - threshold
    if len(exceedances) < 3:
        print(f"{company}: Too few exceedances, skipping")
        continue

    # GPD fit
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    print(f"{company}: Shape={shape:.4f}, Scale={scale:.4f}, Threshold={threshold:.4f}")
    
    exceed_dict[company] = {
        'threshold': threshold,
        'shape': shape,
        'scale': scale,
        'exceedances': exceedances
    }
    
    all_exceed_max = max(all_exceed_max, exceedances.max())

# Prepare meshgrid for 3D surface
x_vals = np.linspace(0, all_exceed_max*1.2, 300)  # slightly extend for smoothness
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Fill Z with GPD PDFs
for i, company in enumerate(y_labels):
    if company not in exceed_dict:
        continue
    shape = exceed_dict[company]['shape']
    scale = exceed_dict[company]['scale']
    Z[i, :] = genpareto.pdf(x_vals, shape, loc=0, scale=scale)

# 3D Plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(X, Y, Z, color='black', alpha=0.3)

ax.set_xlabel('Excess over Threshold')
ax.set_ylabel('Company')
ax.set_yticks(y_vals)
ax.set_yticklabels(y_labels)
ax.set_zlabel('Density')
ax.set_title('Joint POT GPD PDFs for Ford, GM, Stellantis')

plt.show()
