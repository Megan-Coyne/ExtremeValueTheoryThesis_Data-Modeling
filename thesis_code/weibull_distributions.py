import pandas as pd
from Weibull.weibull import fit_weibull
# from GoodnessOfFit.KStest import ks_test, plot_ks_test

df = pd.read_csv("data/FordStock_2015.csv", parse_dates=["date"], index_col="date")
data = df['RET'].abs()

weibull_results = fit_weibull(data, company="Ford", years=[1990,1995,2000,2005,2010], shift_log=True)

print("Overall Weibull Shape:", weibull_results["overall"]["shape"])
