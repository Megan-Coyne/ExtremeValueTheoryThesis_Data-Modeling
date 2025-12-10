import pandas as pd
from GEV.generalized_extreme_value import fit_gev_block_maxima, plot_gev_fit
from GoodnessOfFit.KStest import ks_test, plot_ks_test


# ------------------------------------
# Ford
# ------------------------------------
df = pd.read_csv("data/FordStock_2015.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
data = df["RET"].dropna()

ford_fit = fit_gev_block_maxima(data, company="Ford", block_freq="ME", shift_log=True)

print("Shape k:", ford_fit["shape"])
print("Location μ:", ford_fit["loc"])
print("Scale σ:", ford_fit["scale"])

plot_gev_fit(ford_fit)

ks_result = ks_test(ford_fit["block_maxima"], ford_fit["gev_dist"])
plot_ks_test(ks_result, title="Ford GEV KS Test")


# ------------------------------------
# GM
# ------------------------------------
df = pd.read_csv("data/GMStock_2015.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
data = df["RET"].dropna()

gm_fit = fit_gev_block_maxima(data, company="General Motors", block_freq="ME", shift_log=True)

print("Shape k:", gm_fit["shape"])
print("Location μ:", gm_fit["loc"])
print("Scale σ:", gm_fit["scale"])

plot_gev_fit(gm_fit)

ks_result = ks_test(gm_fit["block_maxima"], gm_fit["gev_dist"])
plot_ks_test(ks_result, title="General Motors GEV KS Test")


# ------------------------------------
# Stellantis/Chrysler
# ------------------------------------
df = pd.read_csv("data/StellantisStock_2015.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
data = df["RET"].dropna()

chrysler_fit = fit_gev_block_maxima(data, company="Chrysler", block_freq="ME", shift_log=True)

print("Shape k:", chrysler_fit["shape"])
print("Location μ:", chrysler_fit["loc"])
print("Scale σ:", chrysler_fit["scale"])

plot_gev_fit(chrysler_fit)

ks_result = ks_test(chrysler_fit["block_maxima"], chrysler_fit["gev_dist"])
plot_ks_test(ks_result, title="Chrysler GEV KS Test")