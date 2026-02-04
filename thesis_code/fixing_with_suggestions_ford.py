import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from GPD.generalized_pareto import fit_gpd  
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# FUNCTION: DAILY EXTREME SCORE
# -------------------------------
def daily_extreme_score(data, fit_result, tail='upper'):
    threshold = fit_result["threshold"]
    gpd_dist = fit_result["gpd_dist"]
    data = np.asarray(data)
    scores = np.zeros_like(data, dtype=float)

    if tail == 'upper':
        exceedances = data - threshold
        exceedances[exceedances < 0] = 0
        scores = np.where(exceedances > 0, gpd_dist.sf(exceedances), 0.0)
    elif tail == 'lower':
        exceedances = threshold - data
        exceedances[exceedances < 0] = 0
        scores = np.where(exceedances > 0, gpd_dist.sf(exceedances), 0.0)
    elif tail == 'both':
        exceedances = np.abs(data - threshold)
        exceedances[exceedances < 0] = 0
        scores = np.where(exceedances > 0, gpd_dist.sf(exceedances), 0.0)
    else:
        raise ValueError("tail must be 'upper', 'lower', or 'both'")
    
    return scores

# -------------------------------
# READ AND PREPARE DATA
# -------------------------------
df = pd.read_csv("data/FordStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Basic features
df['abs_RET'] = df['RET'].abs()
df['log_volume'] = np.log1p(df['VOL'])
df['rel_spread'] = (df['ASK'] - df['BID']) / ((df['ASK'] + df['BID']) / 2)

# -------------------------------
# TRAINING WINDOW FOR EVT FIT
# -------------------------------
train_end = pd.to_datetime('2012-12-31')
train_df = df[df['date'] <= train_end]

quantile = 0.95
# Fit GPD only on training data 
abs_fit = fit_gpd(train_df['abs_RET'].dropna(), "abs_RET", quantile=quantile)
upper_fit = fit_gpd(train_df['RET'].dropna(), "RET_upper", quantile=quantile)
lower_fit = fit_gpd((-train_df['RET']).dropna(), "RET_lower", quantile=quantile)

# -------------------------------
# CALCULATE EXTREME SCORES AND EXCEEDANCES
# -------------------------------
df['abs_extreme_score'] = daily_extreme_score(df['abs_RET'], abs_fit)
df['upper_extreme_score'] = daily_extreme_score(df['RET'], upper_fit)
df['lower_extreme_score'] = daily_extreme_score(-df['RET'], lower_fit)

df['abs_exceed'] = (df['abs_RET'] > abs_fit['threshold']).astype(int)
df['upper_exceed'] = (df['RET'] > upper_fit['threshold']).astype(int)
df['lower_exceed'] = ((-df['RET']) > lower_fit['threshold']).astype(int)

# -------------------------------
# QUARTERLY AGGREGATION
# -------------------------------
agg_dict = {
    'RET': ['mean', 'std', 'max'],
    'abs_RET': ['mean', 'max'],
    'abs_exceed': 'sum',
    'upper_exceed': 'sum',
    'lower_exceed': 'sum',
    'abs_extreme_score': 'mean',
    'upper_extreme_score': 'mean',
    'lower_extreme_score': 'mean',
    'log_volume': ['mean', 'std', 'max'],
    'rel_spread': ['mean', 'max']
}
df_q = df.set_index('date').resample('QE').agg(agg_dict)
df_q.columns = ['_'.join(col).strip() for col in df_q.columns.values]
df_q = df_q.reset_index().rename(columns={'date': 'quarter_end'})

# Days per quarter
df_q['days_in_quarter'] = df.set_index('date').resample('QE').size().values

spread_q95 = train_df['rel_spread'].quantile(0.95)
vol_q05 = train_df['log_volume'].quantile(0.05)
vol_q95 = train_df['log_volume'].quantile(0.95)

# Liquidity ratios
df_q['high_spread_ratio'] = (df.set_index('date')['rel_spread'] > spread_q95).resample('QE').mean().values
df_q['low_volume_ratio'] = (df.set_index('date')['log_volume'] < vol_q05).resample('QE').mean().values
df_q['high_volume_ratio'] = (df.set_index('date')['log_volume'] > vol_q95).resample('QE').mean().values

# EVT ratios
df_q['abs_exceed_ratio'] = df_q['abs_exceed_sum'] / df_q['days_in_quarter']
df_q['upper_exceed_ratio'] = df_q['upper_exceed_sum'] / df_q['days_in_quarter']
df_q['lower_exceed_ratio'] = df_q['lower_exceed_sum'] / df_q['days_in_quarter']

# -------------------------------
# CALCULATE QUARTERLY EVT METRICS (MEAN EXCESS, MAX, VaR, ES)
# -------------------------------
abs_mean_excess, abs_max_excess, abs_var, abs_es_q = [], [], [], []
upper_mean_excess, upper_max_excess, upper_var, upper_es_q = [], [], [], []
lower_mean_excess, lower_max_excess, lower_var, lower_es_q = [], [], [], []

for q, group in df.set_index('date').resample('QE'):
    # ABSOLUTE
    exceed = group['abs_RET'] - abs_fit['threshold']
    exceed = exceed[exceed > 0]
    abs_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    abs_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    var_q = abs_fit['threshold'] + abs_fit['gpd_dist'].ppf(0.99)
    abs_var.append(var_q)
    xi, sigma, threshold = abs_fit['xi'], abs_fit['sigma'], abs_fit['threshold']
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    abs_es_q.append(es_q)

    # UPPER
    exceed = group['RET'] - upper_fit['threshold']
    exceed = exceed[exceed > 0]
    upper_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    upper_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    var_q = upper_fit['threshold'] + upper_fit['gpd_dist'].ppf(0.99)
    upper_var.append(var_q)
    xi, sigma, threshold = upper_fit['xi'], upper_fit['sigma'], upper_fit['threshold']
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    upper_es_q.append(es_q)

    # LOWER
    exceed = (-group['RET']) - lower_fit['threshold']
    exceed = exceed[exceed > 0]
    lower_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    lower_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    var_q = lower_fit['threshold'] + lower_fit['gpd_dist'].ppf(0.99)
    lower_var.append(var_q)
    xi, sigma, threshold = lower_fit['xi'], lower_fit['sigma'], lower_fit['threshold']
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    lower_es_q.append(es_q)

df_q['abs_mean_excess'] = abs_mean_excess
df_q['abs_max_excess'] = abs_max_excess
df_q['abs_var'] = abs_var
df_q['abs_es_q'] = abs_es_q

df_q['upper_mean_excess'] = upper_mean_excess
df_q['upper_max_excess'] = upper_max_excess
df_q['upper_var'] = upper_var
df_q['upper_es_q'] = upper_es_q

df_q['lower_mean_excess'] = lower_mean_excess
df_q['lower_max_excess'] = lower_max_excess
df_q['lower_var'] = lower_var
df_q['lower_es_q'] = lower_es_q

# -------------------------------
# TARGET VARIABLE
# -------------------------------
df_q['municipal_decline'] = (df_q['quarter_end'] >= pd.to_datetime('2013-07-01')).astype(int)

# -------------------------------
# FINAL FEATURE SET
# -------------------------------
# features = [
#     # Absolute tail
#     'abs_mean_excess', 'abs_var', 'abs_es_q', 'abs_exceed_ratio', 'abs_extreme_score_mean',
#     # Upper tail
#     'upper_mean_excess', 'upper_var', 'upper_es_q', 'upper_exceed_ratio',
#     # Lower tail
#     'lower_mean_excess', 'lower_var', 'lower_es_q', 'lower_exceed_ratio',
#     # Liquidity & stock
#     'RET_mean', 'RET_std', 'rel_spread_mean', 'log_volume_mean'
# ]
# features = [
#     'abs_exceed_ratio',
#     'abs_mean_excess',
#     # 'lower_exceed_ratio',
#     'lower_mean_excess',
#     'rel_spread_mean',
#     'abs_es_q'
# ]

features = [
    # EVT 
    'abs_exceed_ratio',
    'abs_mean_excess',
    'abs_es_q',

    # 'lower_mean_excess',

    # Volume liquidity
    'log_volume_mean',
    'log_volume_std',
    'low_volume_ratio',

    # Spread liquidity
    'rel_spread_mean'
    # 'rel_spread_max',
    # 'high_spread_ratio'
]

df_q[features] = df_q[features].fillna(0.0)
df_q[features] = df_q[features].replace([np.inf, -np.inf], 0)

# -------------------------------
# CORRELATION CHECK
# -------------------------------
corr_matrix = df_q[features].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# # Drop features with |ρ|>0.9 with others in same group to reduce multicollinearity
# high_corr_pairs = [(f1,f2) for f1 in features for f2 in features if f1!=f2 and abs(corr_matrix.loc[f1,f2])>0.9]
# drop_features = list(set([f2 for f1,f2 in high_corr_pairs]))  # keep first, drop second
# df_q = df_q.drop(columns=drop_features)
# features = [f for f in features if f not in drop_features]

# print(f"Dropped features due to high correlation: {drop_features}")

# -------------------------------
# MODEL PREPARATION
# -------------------------------
X = df_q[features].values
y = df_q['municipal_decline'].values
indices = np.arange(len(df_q))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y
)

# split_date = pd.to_datetime('2012-10-01')

# train_mask = df_q['quarter_end'] < split_date
# test_mask  = df_q['quarter_end'] >= split_date

# X_train = X[train_mask]
# X_test  = X[test_mask]
# y_train = y[train_mask]
# y_test  = y[test_mask]

# idx_train = df_q.index[train_mask]
# idx_test  = df_q.index[test_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------------
# LOGISTIC REGRESSION WITH L1 REGULARIZATION
# -------------------------------
clf = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', max_iter=1000)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:,1]

print("Test set classification report:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC (test):", roc_auc_score(y_test, y_prob))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_pred))

coef = clf.coef_[0]
for f, c in sorted(zip(features, coef), key=lambda x: -abs(x[1])):
    print(f"{f:30s} coef={c:.4f}")

# -------------------------------
# ANALYZE PREDICTED MUNICIPAL DECLINE DATES
# -------------------------------
df_q['pred_prob'] = np.nan
df_q.loc[idx_test, 'pred_prob'] = y_prob
df_q.loc[idx_train, 'pred_prob'] = clf.predict_proba(X_train_s)[:,1]

decline_threshold = 0.5
predicted_decline_dates = df_q.loc[df_q['pred_prob'] >= decline_threshold, 'quarter_end'].apply(lambda x: x.date()).tolist()
actual_decline_dates = df_q.loc[df_q['municipal_decline']==1, 'quarter_end'].apply(lambda x: x.date()).tolist()
overlap_decline_dates = df_q.loc[(df_q['municipal_decline']==1) & (df_q['pred_prob']>=decline_threshold), 'quarter_end'].apply(lambda x: x.date()).tolist()

tp = len(overlap_decline_dates)
fp = len(predicted_decline_dates) - tp
fn = len(actual_decline_dates) - tp

print("\n--- Municipal Decline Date Analysis ---")
print(f"Total Quarters: {len(df_q)}")
print(f"Actual Decline Quarters: {len(actual_decline_dates)}")
print(f"Predicted Decline Quarters: {len(predicted_decline_dates)}")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

# -------------------------------
# SAVE QUARTER-LEVEL METRICS + PREDICTIONS
# -------------------------------

# Binary prediction using same threshold as analysis
df_q['pred_decline'] = (df_q['pred_prob'] >= decline_threshold).astype(int)

core_cols = [
    'quarter_end',
    'municipal_decline',
    'pred_prob',
    'pred_decline'
]

other_cols = [c for c in df_q.columns if c not in core_cols]
df_q_out = df_q[core_cols + other_cols]

# Save to CSV
output_path = "quarterly_evt_metrics_and_decline_probabilities_ford.csv"
df_q_out.to_csv(output_path, index=False)

print(f"\nSaved quarterly metrics and decline probabilities to:\n{output_path}")
