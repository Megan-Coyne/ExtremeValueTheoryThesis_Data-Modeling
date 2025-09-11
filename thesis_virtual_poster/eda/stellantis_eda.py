import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Stellantis.csv', parse_dates=[0], index_col=0)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_index()

print(df.index.min(), df.index.max())
print(df.head())
print(df.tail())

df_filtered = df.loc['1991-01-31':'2010-10-29']

plt.figure(figsize=(14, 6))
plt.plot(df_filtered.index, df_filtered['Price'], label='Price', color='navy')
plt.title('Stellantis (STLA) Closing Price (1991–2010)')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.legend()
plt.show()
