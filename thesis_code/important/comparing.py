# verison 1

# # -------------------------------
# # IMPORTS
# # -------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
# import statsmodels.api as sm

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# evt_file = "quarterly_evt_metrics_and_decline_probabilities_ford.csv"
# econ_file = "data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv"

# df_evt = pd.read_csv(evt_file)
# df_evt['quarter_end'] = pd.to_datetime(df_evt['quarter_end']).dt.normalize()  # remove time

# df_econ = pd.read_csv(econ_file)
# df_econ['observation_date'] = pd.to_datetime(df_econ['observation_date'])

# # -------------------------------
# # CONVERT MONTHLY ECONOMIC DATA TO QUARTERS
# # -------------------------------
# df_econ['quarter_end'] = df_econ['observation_date'].dt.to_period('Q').dt.end_time
# df_econ_quarterly = df_econ.groupby('quarter_end')['DWLAGRIDX'].mean().reset_index()
# df_econ_quarterly['quarter_end'] = pd.to_datetime(df_econ_quarterly['quarter_end']).dt.normalize()

# # -------------------------------
# # MERGE EVT DATA WITH ECONOMIC INDEX
# # -------------------------------
# df_merged = pd.merge(df_evt, df_econ_quarterly, on='quarter_end', how='inner')

# # -------------------------------
# # DEBUG INFO
# # -------------------------------
# print("EVT date range:", df_evt['quarter_end'].min(), "→", df_evt['quarter_end'].max())
# print("Economic index date range:", df_econ_quarterly['quarter_end'].min(), "→", df_econ_quarterly['quarter_end'].max())
# print("Merged rows:", len(df_merged))
# print(df_merged[['quarter_end', 'DWLAGRIDX', 'pred_decline']].head())

# # -------------------------------
# # PEARSON CORRELATION FOR BINARY pred_decline
# # -------------------------------
# corr, pval = pearsonr(df_merged['pred_decline'], df_merged['DWLAGRIDX'])
# print(f"\nPearson correlation of pred_decline with Economic Index: r={corr:.4f}, p={pval:.4f}")

# # -------------------------------
# # LAGGED LOGISTIC REGRESSION
# # -------------------------------
# # Create lagged economic index (1 quarter)
# df_merged_sorted = df_merged.sort_values('quarter_end')
# df_merged_sorted['econ_lag1'] = df_merged_sorted['DWLAGRIDX'].shift(1)
# df_logit = df_merged_sorted.dropna(subset=['econ_lag1', 'pred_decline'])

# X = sm.add_constant(df_logit['econ_lag1'])  # add intercept
# y = df_logit['pred_decline']

# logit_model = sm.Logit(y, X).fit(disp=False)
# print("\nLagged Logistic Regression (pred_decline ~ previous quarter DWLAGRIDX):")
# print(logit_model.summary())

# # -------------------------------
# # PLOT RELATIONSHIPS
# # -------------------------------
# plt.figure(figsize=(10,6))
# sns.scatterplot(data=df_merged, x='DWLAGRIDX', y='pred_decline', alpha=0.6)
# plt.title("Predicted Municipal Decline vs Economic Index")
# plt.xlabel("Economic Conditions Index (Quarterly Average)")
# plt.ylabel("Predicted Decline (0/1)")
# plt.show()

# version 2

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Read EVT files
# -------------------------------
ford = pd.read_csv('quarterly_evt_metrics_and_decline_probabilities_ford.csv')
gm = pd.read_csv('quarterly_evt_metrics_and_decline_probabilities_gm.csv')
chrysler = pd.read_csv('quarterly_evt_metrics_and_decline_probabilities_chrysler.csv')

# Make sure the date/quarter column is datetime or period
for df in [ford, gm, chrysler]:
    df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')

# -------------------------------
# 2️⃣ Read economic index
# -------------------------------
econ = pd.read_csv('data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv')  # with columns 'observation_date', 'DWLAGRIDX'
econ['observation_date'] = pd.to_datetime(econ['observation_date'])
econ['quarter'] = econ['observation_date'].dt.to_period('Q')

# Aggregate monthly index to quarterly average
econ_quarterly = econ.groupby('quarter')['DWLAGRIDX'].mean().reset_index()

# -------------------------------
# 3️⃣ Merge economic index with EVT data
# -------------------------------
ford = ford.merge(econ_quarterly, on='quarter', how='left')
gm = gm.merge(econ_quarterly, on='quarter', how='left')
chrysler = chrysler.merge(econ_quarterly, on='quarter', how='left')

# -------------------------------
# 4️⃣ Correlation analysis
# -------------------------------
print("Ford correlation with DWLAGRIDX:", ford['decline_prob'].corr(ford['DWLAGRIDX']))
print("GM correlation with DWLAGRIDX:", gm['decline_prob'].corr(gm['DWLAGRIDX']))
print("Chrysler correlation with DWLAGRIDX:", chrysler['decline_prob'].corr(chrysler['DWLAGRIDX']))

# -------------------------------
# 5️⃣ Simple regression: decline probability ~ economic index
# -------------------------------
def run_regression(df, company_name):
    X = sm.add_constant(df['DWLAGRIDX'])
    y = df['decline_prob']
    model = sm.OLS(y, X).fit()
    print(f"\n--- Regression for {company_name} ---")
    print(model.summary())
    return model

ford_model = run_regression(ford, 'Ford')
gm_model = run_regression(gm, 'GM')
chrysler_model = run_regression(chrysler, 'Chrysler')

# -------------------------------
# 6️⃣ Plot decline probability vs economic index
# -------------------------------
def plot_decline_vs_index(df, company_name):
    plt.figure(figsize=(6,4))
    plt.scatter(df['DWLAGRIDX'], df['decline_prob'], alpha=0.7)
    plt.xlabel('Economic Index (DWLAGRIDX)')
    plt.ylabel('Decline Probability')
    plt.title(f'{company_name} Decline Probability vs Economic Index')
    plt.grid(True)
    plt.show()

plot_decline_vs_index(ford, 'Ford')
plot_decline_vs_index(gm, 'GM')
plot_decline_vs_index(chrysler, 'Chrysler')

# -------------------------------
# 7️⃣ Optional: Combine all automakers
# -------------------------------
all_automakers = pd.concat([
    ford.assign(company='Ford'),
    gm.assign(company='GM'),
    chrysler.assign(company='Chrysler')
])

# Correlation for pooled dataset
print("Pooled correlation:", all_automakers['decline_prob'].corr(all_automakers['DWLAGRIDX']))

# Pooled regression with company as categorical variable
all_automakers = pd.get_dummies(all_automakers, columns=['company'], drop_first=True)
X = sm.add_constant(all_automakers[['DWLAGRIDX', 'company_GM', 'company_Chrysler']])
y = all_automakers['decline_prob']
pooled_model = sm.OLS(y, X).fit()
print("\n--- Pooled Regression ---")
print(pooled_model.summary())
