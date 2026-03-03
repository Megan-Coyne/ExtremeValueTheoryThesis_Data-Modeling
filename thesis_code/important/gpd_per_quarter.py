# -------------------------------
# QUARTERLY EVT WITH QUARTER-SPECIFIC GPD
# -------------------------------

import pandas as pd
import numpy as np
from datetime import datetime

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
# FUNCTION: QUARTERLY GPD METRICS
# -------------------------------
def quarterly_gpd_metrics(series, quantile=0.95):
    """
    Fit GPD to one quarter of data and compute EVT metrics.
    Returns dictionary of metrics + parameters.
    """

    series = series.dropna()
    fit = fit_gpd(series, "quarter_fit", quantile=quantile)

    xi = fit["xi"]
    sigma = fit["sigma"]
    threshold = fit["threshold"]

    exceed = series - threshold
    exceed = exceed[exceed > 0]

    if len(exceed) == 0 or xi >= 1:
        es_q = 0.0
        mean_excess = 0.0
        max_excess = 0.0
    else:
        mean_excess = exceed.mean()
        max_excess = exceed.max()
        y_q = max_excess
        es_q = threshold + (y_q + sigma - xi*y_q) / (1 - xi)

    return {
        "mean_excess": mean_excess,
        "max_excess": max_excess,
        "es_q": es_q,
        "xi": xi,
        "sigma": sigma,
        "threshold": threshold
    }

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
# Fit GPD on training data for daily extreme scores only
abs_fit = fit_gpd(train_df['abs_RET'].dropna(), "abs_RET", quantile=quantile)
upper_fit = fit_gpd(train_df['RET'].dropna(), "RET_upper", quantile=quantile)
lower_fit = fit_gpd((-train_df['RET']).dropna(), "RET_lower", quantile=quantile)

# -------------------------------
# DAILY EXTREME SCORES
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
# CALCULATE QUARTERLY EVT METRICS (MEAN EXCESS, MAX, CONDITIONAL ES) WITH QUARTER-SPECIFIC FITS
# -------------------------------
results = []

for q, group in df.set_index('date').resample('QE'):

    abs_metrics = quarterly_gpd_metrics(group['abs_RET'])
    upper_metrics = quarterly_gpd_metrics(group['RET'])
    lower_metrics = quarterly_gpd_metrics(-group['RET'])

    results.append({
        "quarter_end": q,

        # ABSOLUTE
        "abs_mean_excess": abs_metrics["mean_excess"],
        "abs_max_excess": abs_metrics["max_excess"],
        "abs_es_q": abs_metrics["es_q"],
        "abs_xi": abs_metrics["xi"],
        "abs_sigma": abs_metrics["sigma"],
        "abs_threshold": abs_metrics["threshold"],

        # UPPER
        "upper_mean_excess": upper_metrics["mean_excess"],
        "upper_max_excess": upper_metrics["max_excess"],
        "upper_es_q": upper_metrics["es_q"],
        "upper_xi": upper_metrics["xi"],
        "upper_sigma": upper_metrics["sigma"],
        "upper_threshold": upper_metrics["threshold"],

        # LOWER
        "lower_mean_excess": lower_metrics["mean_excess"],
        "lower_max_excess": lower_metrics["max_excess"],
        "lower_es_q": lower_metrics["es_q"],
        "lower_xi": lower_metrics["xi"],
        "lower_sigma": lower_metrics["sigma"],
        "lower_threshold": lower_metrics["threshold"],
    })

df_q_metrics = pd.DataFrame(results)
df_q = df_q.merge(df_q_metrics, on='quarter_end', how='left')

# -------------------------------
# TARGET VARIABLE
# -------------------------------
df_q['municipal_decline'] = (df_q['quarter_end'] >= pd.to_datetime('2013-07-01')).astype(int)

# -------------------------------
# FINAL FEATURE SET
# -------------------------------
features = [
    # EVT 
    'abs_es_q',
    'abs_exceed_ratio',
    'abs_xi',
    'abs_sigma',

    # Volume liquidity
    'log_volume_mean',
    'log_volume_std',
    # 'low_volume_ratio',

    # Spread liquidity
    'rel_spread_mean'
    # 'abs_mean_excess'
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

# -------------------------------
# MODEL PREPARATION
# -------------------------------
X = df_q[features].values
y = df_q['municipal_decline'].values
indices = np.arange(len(df_q))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y
)

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
df_q['pred_decline'] = (df_q['pred_prob'] >= decline_threshold).astype(int)

core_cols = [
    'quarter_end',
    'municipal_decline',
    'pred_prob',
    'pred_decline'
]

other_cols = [c for c in df_q.columns if c not in core_cols]
df_q_out = df_q[core_cols + other_cols]

output_path = "quarterly_evt_metrics_and_decline_probabilities_ford.csv"
df_q_out.to_csv(output_path, index=False)
print(f"\nSaved quarterly metrics and decline probabilities to:\n{output_path}")

# -------------------------------
# SAVE TRAINING AND TEST SETS
# -------------------------------
train_df_q = df_q.loc[idx_train].reset_index(drop=True)
test_df_q = df_q.loc[idx_test].reset_index(drop=True)

train_output_path = "quarterly_evt_train_ford.csv"
test_output_path = "quarterly_evt_test_ford.csv"

train_df_q.to_csv(train_output_path, index=False)
test_df_q.to_csv(test_output_path, index=False)

print(f"\nSaved training set to: {train_output_path}")
print(f"Saved test set to: {test_output_path}")
