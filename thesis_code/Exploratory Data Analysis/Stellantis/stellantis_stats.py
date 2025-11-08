import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/StellantisStock.csv")  # replace with your file path

# Columns to analyze
numeric_cols = ['BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET', 'BID', 'ASK', 'OPENPRC', 'RETX']
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

# Format numbers nicely
def format_value(col):
    if col.name in ['RET', 'RETX']:
        return col.apply(lambda x: f"{x:.4f}")
    elif col.name == 'VOL':
        return col.apply(lambda x: f"{int(x):,}")
    else:
        return col.apply(lambda x: f"{x:.2f}")

summary_formatted = summary.copy()
summary_formatted['Mean'] = format_value(summary['Mean'])
summary_formatted['Standard Deviation'] = format_value(summary['Standard Deviation'])

# --- Create a figure for the table ---
fig, ax = plt.subplots(figsize=(10, 4))
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

plt.title("Summary Statistics of Stellantis Stock Data", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
