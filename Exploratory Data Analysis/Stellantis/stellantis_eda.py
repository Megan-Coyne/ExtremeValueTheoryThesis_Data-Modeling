import pandas as pd

# LOAD DATA
df = pd.read_csv("data/StellantisStock.csv", parse_dates=["date"])

# CHECK BASIC INFO ABOUT THE DATASET
print(df.head())       # first few rows
print(df.info())       # column types & non-null counts
print(df.describe())   # summary statistics

# FIX NEGATIVE PRICES (CRSP SOMETIMES STORES PRC AS NEGATIVE FOR SHORT SALES)
df["PRC"] = df["PRC"].abs()

import matplotlib.pyplot as plt

# PLOT Stellantis STOCK PRICE OVER TIME
plt.figure(num="Stellantis Stock Price (1990–2024)", figsize=(12,6))
plt.plot(df["date"], df["PRC"])
plt.title("Stellantis Stock Price (1990–2024)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

# PLOT TRADING VOLUME OVER TIME
plt.figure(num="Stellantis Trading Volume (1990–2024)", figsize=(12,4))
plt.plot(df["date"], df["VOL"])
plt.title("Stellantis Trading Volume (1990–2024)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

import numpy as np

# CONVERT RETURNS TO NUMERIC (SOMETIMES RETURNS ARE STRINGS OR HAVE MISSING VALUES)
df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
df["RETX"] = pd.to_numeric(df["RETX"], errors="coerce")

# CUMULATIVE RETURNS (GROWTH OF $1 INVESTED)
df["cum_return"] = (1 + df["RETX"].fillna(0)).cumprod()

plt.figure(num="Stellantis Cumulative Return (1990–2024)", figsize=(12,6))
plt.plot(df["date"], df["cum_return"])
plt.title("Stellantis Cumulative Return (1990–2024)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 investment")
plt.show()

# DISTRIBUTION OF DAILY RETURNS
plt.figure(num="Stellantis Distribution of Daily Returns", figsize=(8,5))
plt.hist(df["RETX"].dropna(), bins=100, alpha=0.7)
plt.title("Distribution of Stellantis Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

# SET DATE AS INDEX FOR RESAMPLING & ROLLING CALCULATIONS
df.set_index("date", inplace=True)

# MOVING AVERAGES & ROLLING VOLATILITY
df["MA50"] = df["PRC"].rolling(50).mean()       # 50-day moving average
df["MA200"] = df["PRC"].rolling(200).mean()     # 200-day moving average
df["volatility_30d"] = df["RETX"].rolling(30).std()  # 30-day rolling standard deviation

# PLOT STOCK PRICE WITH MOVING AVERAGES
plt.figure(num="Stellantis Stock Price with Moving Averages", figsize=(12,6))
plt.plot(df.index, df["PRC"], label="Price")
plt.plot(df.index, df["MA50"], label="50-day MA")
plt.plot(df.index, df["MA200"], label="200-day MA")
plt.legend()
plt.title("Stellantis Stock with Moving Averages")
plt.show()

# PLOT 30-DAY ROLLING VOLATILITY
plt.figure(num="Stellantis 30-Day Rolling Volatility", figsize=(12,4))
plt.plot(df.index, df["volatility_30d"])
plt.title("Stellantis 30-Day Rolling Volatility")
plt.show()

# RESAMPLE RETURNS TO MONTHLY & YEARLY AVERAGES
monthly = df["RETX"].resample("ME").mean()
yearly = df["RETX"].resample("YE").mean()
# Convert index to just the year
yearly.index = yearly.index.year

# PLOT AVERAGE YEARLY RETURNS
plt.figure(num="Stellantis Average Yearly Returns", figsize=(12,6))
yearly.plot(kind="bar")
plt.title("Stellantis Avg Yearly Returns")
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.show()

# PLOT AVERAGE MONTHLY VOLUME
monthly_vol = df["VOL"].resample("ME").mean()
plt.figure(num="Stellantis Average Monthly Volume", figsize=(12,4))
monthly_vol.plot(title="Stellantis Average Monthly Trading Volume")
plt.show()