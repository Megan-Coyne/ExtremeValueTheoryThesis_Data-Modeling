import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Files and labels
files = {
    'Ford': 'data/FordStock.csv',
    'GM': 'data/GMStock.csv',
    'Stellantis': 'data/StellantisStock.csv'
}

# Prepare a grid for the 3D surface
x_vals = np.linspace(0, 0.1, 200)  # Adjust max depending on your returns
y_labels = list(files.keys())
y_vals = np.arange(len(y_labels))  # 0, 1, 2

X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Compute PDF for each company
for i, company in enumerate(y_labels):
    df = pd.read_csv(files[company])
    df['date'] = pd.to_datetime(df['date'])
    df['abs_RET'] = df['RET'].abs().dropna()
    data = df['abs_RET'].dropna()
    
    shape, loc, scale = weibull_min.fit(data, floc=0)
    print(f"{company}: Shape = {shape:.4f}, Scale = {scale:.4f}")
    
    # Fill Z values for this company across X
    Z[i, :] = weibull_min.pdf(x_vals, shape, loc=0, scale=scale)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Optional: scatter for clarity
ax.scatter(X, Y, Z, color='black', alpha=0.3)

# Labels
ax.set_xlabel('Absolute Returns')
ax.set_ylabel('Company')
ax.set_yticks(y_vals)
ax.set_yticklabels(y_labels)
ax.set_zlabel('Density')
ax.set_title('Joint Weibull PDFs for Ford, GM, Stellantis')

plt.show()
