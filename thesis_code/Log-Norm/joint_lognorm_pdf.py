import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from mpl_toolkits.mplot3d import Axes3D

# Files and labels
files = {
    'Ford': 'data/FordStock.csv',
    'GM': 'data/GMStock.csv',
    'Stellantis': 'data/StellantisStock.csv'
}

y_labels = list(files.keys())
y_vals = np.arange(len(y_labels))

# Fit log-normal for each company
lognorm_dict = {}
all_data_max = 0

for company in y_labels:
    df = pd.read_csv(files[company])
    df['date'] = pd.to_datetime(df['date'])
    df['abs_RET'] = df['RET'].abs()
    data = df['abs_RET'].dropna()

    # Shift slightly if any zeros
    data_shift = data + 1e-6 if np.any(data <= 0) else data

    # Fit log-normal (fix loc=0)
    shape, loc, scale = lognorm.fit(data_shift, floc=0)
    print(f"{company}: Shape={shape:.4f}, Scale={scale:.4f}, Location={loc:.4f}")

    lognorm_dict[company] = {
        'shape': shape,
        'scale': scale,
        'loc': loc,
        'data': data_shift
    }

    all_data_max = max(all_data_max, data_shift.max())

# Prepare meshgrid
x_vals = np.linspace(0, all_data_max*1.2, 300)  # extend slightly
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Fill Z with log-normal PDFs
for i, company in enumerate(y_labels):
    shape = lognorm_dict[company]['shape']
    scale = lognorm_dict[company]['scale']
    loc = lognorm_dict[company]['loc']
    Z[i, :] = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)

# 3D Surface Plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(X, Y, Z, color='black', alpha=0.3, s=5)

ax.set_xlabel('Absolute Returns')
ax.set_ylabel('Company')
ax.set_yticks(y_vals)
ax.set_yticklabels(y_labels)
ax.set_zlabel('Density')
ax.set_title('Joint Log-Normal PDFs for Ford, GM, Stellantis')

plt.show()
