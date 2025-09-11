import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ford.csv', index_col=0)

df.index = pd.to_datetime(df.index, errors='coerce')

df = df[df.index.notna()]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Close'])

df = df.sort_index()

df_filtered = df.loc['1991-01-31':'2013-10-29']

plt.figure(figsize=(14, 6))
plt.plot(df_filtered.index, df_filtered['Close'], label='Close', color='navy')
plt.title('Ford (F) Closing Price (1991–2010)')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.legend()
plt.show()
