import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv")  # replace with your file path

# Column to analyze
numeric_cols = ['DWLAGRIDX']
data = df[numeric_cols].copy()

# Convert to numeric
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaNs
data = data.dropna()

# Compute mean and standard deviation
means = data.mean()
stds = data.std()

# Combine into a summary table
summary = pd.DataFrame({
    'Mean': means,
    'Standard Deviation': stds
})

# Format numbers nicely (2 decimal places for your index)
summary_formatted = summary.copy()
summary_formatted['Mean'] = summary['Mean'].apply(lambda x: f"{x:.2f}")
summary_formatted['Standard Deviation'] = summary['Standard Deviation'].apply(lambda x: f"{x:.2f}")

# --- Create a figure for the table ---
fig, ax = plt.subplots(figsize=(6, 2))  # smaller figure for a single column
ax.axis('off')  # hide axes

# Draw table
table = ax.table(
    cellText=summary_formatted.values,
    colLabels=summary_formatted.columns,
    rowLabels=summary_formatted.index,
    cellLoc='center',
    rowLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(summary_formatted.columns))))

plt.title("Summary Statistics of Stellantis DWLAGRIDX", fontsize=12, pad=15)
plt.tight_layout()
plt.show()
