import pandas as pd
import numpy as np

from GPD.generalized_pareto import fit_gpd, plot_gpd_fit
from GoodnessOfFit.KStest import ks_test, plot_ks_test
# from Regression.regression import tail_regression  # <- make sure this is imported

# ------------------------------------
# Ford
# ------------------------------------
df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")

df['date'] = pd.to_datetime(df['date'])
df['abs_RET'] = df['RET'].abs()
overall_data = df['abs_RET'].dropna()

ford_fit = fit_gpd(overall_data, "Ford", quantile=0.95)
print("Shape ξ:", ford_fit["xi"])
print("Scale σ:", ford_fit["sigma"])
print("Threshold:", ford_fit["threshold"])

plot_gpd_fit(overall_data, ford_fit)

# # Tail regression on exceedances
transform = lambda x: x[x > ford_fit["threshold"]] - ford_fit["threshold"]
# regression_result = tail_regression(ford_fit["gpd_dist"], overall_data, tail="upper", quantile=0.95, plot=True)
# print(f"Ford Tail Regression Slope: {regression_result['slope']:.4f}")

# KS test
ks_result = ks_test(overall_data, dist=ford_fit["gpd_dist"], transform=transform)
plot_ks_test(ks_result, title="Ford GPD KS Test")


# # ------------------------------------
# # GM
# # ------------------------------------
# df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/GMStock_2015.csv")
# df['date'] = pd.to_datetime(df['date'])
# df['abs_RET'] = df['RET'].abs()
# overall_data = df['abs_RET'].dropna()

# gm_fit = fit_gpd(overall_data, "General Motors", quantile=0.95)
# print("Shape ξ:", gm_fit["xi"])
# print("Scale σ:", gm_fit["sigma"])
# print("Threshold:", gm_fit["threshold"])

# plot_gpd_fit(overall_data, gm_fit)

# # regression_result = tail_regression(gm_fit["gpd_dist"], overall_data, tail="upper", quantile=0.95, plot=True)
# # print(f"GM Tail Regression Slope: {regression_result['slope']:.4f}")

# transform = lambda x: x[x > gm_fit["threshold"]] - gm_fit["threshold"]
# ks_result = ks_test(overall_data, dist=gm_fit["gpd_dist"], transform=transform)
# plot_ks_test(ks_result, title="GM GPD KS Test")


# # ------------------------------------
# # Stellantis/Chrysler
# # ------------------------------------
# df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/StellantisStock_2015.csv")
# df['date'] = pd.to_datetime(df['date'])
# df['abs_RET'] = df['RET'].abs()
# overall_data = df['abs_RET'].dropna()

# chrysler_fit = fit_gpd(overall_data, "Chrysler", quantile=0.95)
# print("Shape ξ:", chrysler_fit["xi"])
# print("Scale σ:", chrysler_fit["sigma"])
# print("Threshold:", chrysler_fit["threshold"])

# plot_gpd_fit(overall_data, chrysler_fit)

# # regression_result = tail_regression(chrysler_fit["gpd_dist"], overall_data, tail="upper", quantile=0.95, plot=True)
# # print(f"Chrysler Tail Regression Slope: {regression_result['slope']:.4f}")

# transform = lambda x: x[x > chrysler_fit["threshold"]] - chrysler_fit["threshold"]
# ks_result = ks_test(overall_data, dist=chrysler_fit["gpd_dist"], transform=transform)
# plot_ks_test(ks_result, title="Chrysler GPD KS Test")


# ------------------------------------
# Ford - Volume
# ------------------------------------
df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df['abs_VOL'] = df['VOL'].abs()
overall_data = df['abs_VOL'].dropna()

gm_fit = fit_gpd(overall_data, "Ford", quantile=0.95)
print("Shape ξ:", gm_fit["xi"])
print("Scale σ:", gm_fit["sigma"])
print("Threshold:", gm_fit["threshold"])

plot_gpd_fit(overall_data, gm_fit)

# regression_result = tail_regression(gm_fit["gpd_dist"], overall_data, tail="upper", quantile=0.95, plot=True)
# print(f"GM Tail Regression Slope: {regression_result['slope']:.4f}")

transform = lambda x: x[x > gm_fit["threshold"]] - gm_fit["threshold"]
ks_result = ks_test(overall_data, dist=gm_fit["gpd_dist"], transform=transform)
plot_ks_test(ks_result, title="Ford GPD KS Test")