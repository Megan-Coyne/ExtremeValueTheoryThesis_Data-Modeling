import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv", parse_dates=["observation_date"])
df.rename(columns={"observation_date": "date"}, inplace=True)

# CHECK BASICS
print(df.head())       # first rows
print(df.info())       # structure
print(df.describe())   # summary statistics

# PLOT TIME SERIES
plt.figure(num="Detroit Economic Conditions Index", figsize=(12,6))
plt.plot(df["date"], df["DWLAGRIDX"], label="Economic Index")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)  # recession threshold-ish
plt.title("Detroit Economic Conditions Index (1990–2019)")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()

# DISTRIBUTION
plt.figure(num="Distribution of Detroit Index", figsize=(8,5))
plt.hist(df["DWLAGRIDX"].dropna(), bins=30, alpha=0.7)
plt.title("Distribution of Detroit Economic Index")
plt.xlabel("Index Value")
plt.ylabel("Frequency")
plt.show()

# ROLLING MEAN & VOLATILITY
df.set_index("date", inplace=True)
df["MA12"] = df["DWLAGRIDX"].rolling(12).mean()      # 1-year moving average
df["volatility_12m"] = df["DWLAGRIDX"].rolling(12).std()

plt.figure(num="Detroit Index with Moving Average", figsize=(12,6))
plt.plot(df.index, df["DWLAGRIDX"], label="Index")
plt.plot(df.index, df["MA12"], label="12-mo Moving Avg", linewidth=2)
plt.legend()
plt.title("Detroit Economic Conditions Index with 12-Month Moving Average")
plt.show()

plt.figure(num="Detroit Index Rolling Volatility", figsize=(12,4))
plt.plot(df.index, df["volatility_12m"], label="12-mo Std Dev")
plt.title("Detroit Economic Index Rolling Volatility (12 Months)")
plt.show()

# RESAMPLE TO YEARLY
yearly = df["DWLAGRIDX"].resample("Y").mean()
# Convert index to year 
yearly.index = yearly.index.year

plt.figure(num="Yearly Avg Detroit Index", figsize=(12,6))
yearly.plot(kind="bar")
plt.title("Average Yearly Detroit Economic Index")
plt.ylabel("Index Value")
plt.xlabel("Year")
plt.show()
