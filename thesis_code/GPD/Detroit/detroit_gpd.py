import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Load data from file
file_path = "/Users/megancoyne/school/thesis/thesis_code/data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv"
df = pd.read_csv(file_path)

df['observation_date'] = pd.to_datetime(df['observation_date'])
df['abs_DWL'] = df['DWLAGRIDX'].abs()
overall_data = df['abs_DWL'].dropna()

# Function for POT + GPD fit
def pot_gpd_fit(data, threshold_quantile=0.95):
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data[data > threshold] - threshold
    if len(exceedances) < 3:
        return threshold, exceedances, None, None
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return threshold, exceedances, shape, scale

# Overall POT + GPD
threshold, exceedances, shape, scale = pot_gpd_fit(overall_data, threshold_quantile=0.95)

if shape is not None:
    print(f"Overall GPD Fit (POT > {threshold:.4f}) - Shape (ξ): {shape:.4f}, Scale (β): {scale:.4f}")

    # Plot exceedances and GPD fit
    plt.figure(figsize=(10,5))
    plt.hist(exceedances, bins=20, density=True, alpha=0.6, color='skyblue', label='Exceedances')

    x = np.linspace(0, exceedances.max()*1.2, 1000)
    pdf = genpareto.pdf(x, shape, loc=0, scale=scale)
    plt.plot(x, pdf, 'r-', lw=2, label='GPD Fit (POT)')

    plt.xlabel('Exceedances over threshold')
    plt.ylabel('Density')
    plt.title(f'GPD Fit to |DWLAGRIDX| (Overall, Threshold={threshold:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Display MLE Table
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('tight')
    ax.axis('off')

    fit_results = [['Overall', f'{shape:.4f}', f'{scale:.4f}', f'{threshold:.4f}']]
    column_labels = ["Dataset", "Shape (ξ)", "Scale (β)", "Threshold (u)"]

    table = ax.table(cellText=fit_results,
                     colLabels=column_labels,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title("Maximum Likelihood Estimates for GPD Fits (POT) for DWLAGRIDX", pad=20)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough exceedances to fit GPD")
