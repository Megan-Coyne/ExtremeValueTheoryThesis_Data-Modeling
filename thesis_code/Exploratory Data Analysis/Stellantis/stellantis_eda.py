import pandas as pd

# LOAD DATA
df = pd.read_csv("data/StellantisStock_2015.csv", parse_dates=["date"])

# CHECK BASIC INFO ABOUT THE DATASET
print(df.head())       # first few rows
print(df.info())       # column types & non-null counts
print(df.describe())   # summary statistics

# FIX NEGATIVE PRICES (CRSP SOMETIMES STORES PRC AS NEGATIVE FOR SHORT SALES)
df["PRC"] = df["PRC"].abs()

import matplotlib.pyplot as plt

# PLOT Chrysler STOCK PRICE OVER TIME
plt.figure(num="Chrysler Stock Price (1990–2015)", figsize=(12,6))
plt.plot(df["date"], df["PRC"])
plt.title("Chrysler Stock Price (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

# PLOT TRADING VOLUME OVER TIME
plt.figure(num="Chrysler Trading Volume (1990–2015)", figsize=(12,4))
plt.plot(df["date"], df["VOL"])
plt.title("Chrysler Trading Volume (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

import numpy as np

# CONVERT RETURNS TO NUMERIC (SOMETIMES RETURNS ARE STRINGS OR HAVE MISSING VALUES)
df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
df["RETX"] = pd.to_numeric(df["RETX"], errors="coerce")

# CUMULATIVE RETURNS (GROWTH OF $1 INVESTED)
df["cum_return"] = (1 + df["RETX"].fillna(0)).cumprod()

plt.figure(num="Chrysler Cumulative Return (1990–2015)", figsize=(12,6))
plt.plot(df["date"], df["cum_return"])
plt.title("Chrysler Cumulative Return (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 investment")
plt.show()

# DISTRIBUTION OF DAILY RETURNS
plt.figure(num="Chrysler Distribution of Daily Returns", figsize=(8,5))
plt.hist(df["RETX"].dropna(), bins=100, alpha=0.7)
plt.title("Distribution of Chrysler Daily Returns")
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
plt.figure(num="Chrysler Stock Price with Moving Averages", figsize=(12,6))
plt.plot(df.index, df["PRC"], label="Price")
plt.plot(df.index, df["MA50"], label="50-day MA")
plt.plot(df.index, df["MA200"], label="200-day MA")
plt.legend()
plt.title("Chrysler Stock with Moving Averages")
plt.show()

# PLOT 30-DAY ROLLING VOLATILITY
plt.figure(num="Chrysler 30-Day Rolling Volatility", figsize=(12,4))
plt.plot(df.index, df["volatility_30d"])
plt.title("Chrysler 30-Day Rolling Volatility")
plt.show()

# RESAMPLE RETURNS TO MONTHLY & YEARLY AVERAGES
monthly = df["RETX"].resample("ME").mean()
yearly = df["RETX"].resample("YE").mean()
# Convert index to just the year
yearly.index = yearly.index.year

# PLOT AVERAGE YEARLY RETURNS
plt.figure(num="Chrysler Average Yearly Returns", figsize=(12,6))
yearly.plot(kind="bar")
plt.title("Chrysler Avg Yearly Returns")
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.show()

# PLOT AVERAGE MONTHLY VOLUME
monthly_vol = df["VOL"].resample("ME").mean()
plt.figure(num="Chrysler Average Monthly Volume", figsize=(12,4))
monthly_vol.plot(title="Chrysler Average Monthly Trading Volume")
plt.show()