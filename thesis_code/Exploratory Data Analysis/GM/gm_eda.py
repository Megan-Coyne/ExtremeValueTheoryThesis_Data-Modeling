import pandas as pd

df = pd.read_csv("data/GMStock_2015.csv", parse_dates=["date"])

print(df.head())      
print(df.info())     
print(df.describe())   

df["PRC"] = df["PRC"].abs()

import matplotlib.pyplot as plt

plt.figure(num="GM Stock Price (1990–2015)", figsize=(12,6))
plt.plot(df["date"], df["PRC"])
plt.title("GM Stock Price (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

plt.figure(num="GM Trading Volume (1990–2015)", figsize=(12,4))
plt.plot(df["date"], df["VOL"])
plt.title("GM Trading Volume (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

import numpy as np

df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
df["RETX"] = pd.to_numeric(df["RETX"], errors="coerce")

df["cum_return"] = (1 + df["RETX"].fillna(0)).cumprod()

plt.figure(num="GM Cumulative Return (1990–2015)", figsize=(12,6))
plt.plot(df["date"], df["cum_return"])
plt.title("GM Cumulative Return (1990–2015)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 investment")
plt.show()

plt.figure(num="GM Distribution of Daily Returns", figsize=(8,5))
plt.hist(df["RETX"].dropna(), bins=100, alpha=0.7)
plt.title("Distribution of GM Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

df.set_index("date", inplace=True)

df["MA50"] = df["PRC"].rolling(50).mean()       
df["MA200"] = df["PRC"].rolling(200).mean()  
df["volatility_30d"] = df["RETX"].rolling(30).std() 

plt.figure(num="GM Stock Price with Moving Averages", figsize=(12,6))
plt.plot(df.index, df["PRC"], label="Price")
plt.plot(df.index, df["MA50"], label="50-day MA")
plt.plot(df.index, df["MA200"], label="200-day MA")
plt.legend()
plt.title("GM Stock with Moving Averages")
plt.show()

plt.figure(num="GM 30-Day Rolling Volatility", figsize=(12,4))
plt.plot(df.index, df["volatility_30d"])
plt.title("GM 30-Day Rolling Volatility")
plt.show()

monthly = df["RETX"].resample("ME").mean()
yearly = df["RETX"].resample("YE").mean()
yearly.index = yearly.index.year

plt.figure(num="GM Average Yearly Returns", figsize=(12,6))
yearly.plot(kind="bar")
plt.title("GM Avg Yearly Returns")
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.show()

monthly_vol = df["VOL"].resample("ME").mean()
plt.figure(num="GM Average Monthly Volume", figsize=(12,4))
monthly_vol.plot(title="GM Average Monthly Trading Volume")
plt.show()