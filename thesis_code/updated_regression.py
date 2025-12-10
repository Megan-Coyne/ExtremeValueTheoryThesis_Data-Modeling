import pandas as pd
import numpy as np
from datetime import datetime

from GPD.generalized_pareto import fit_gpd, plot_gpd_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['abs_RET'] = df['RET'].abs()

# determine an end date for the training set, specify that the training set is all of the data up to and including that date
train_end = pd.to_datetime('2012-12-31')
train_df = df[df['date'] <= train_end]

# pick the quantile for the gpd threshold, fit the gpd to the absolute returns
# fit the gpd to the highest returns
# fit the gpd to the lowest returns
quantile = 0.95
abs_fit = fit_gpd(train_df['abs_RET'].dropna(), "Ford - abs_RET", quantile=quantile)
upper_fit = fit_gpd(train_df['RET'].dropna(), "Ford - RET (upper)", quantile=quantile)
lower_fit = fit_gpd((-train_df['RET']).dropna(), "Ford - RET (lower via -RET)", quantile=quantile)
# train_abs = np.log1p(train_df['abs_RET'].dropna())
# train_upper = np.log1p(train_df.loc[train_df['RET'] > 0, 'RET'])
# train_lower = np.log1p((-train_df.loc[train_df['RET'] < 0, 'RET']))

# plot_gpd_fit(train_df['abs_RET'].dropna(), abs_fit)
# plot_gpd_fit(train_df['RET'].dropna(), upper_fit)
# plot_gpd_fit((-train_df['RET']).dropna(), lower_fit)

# calculate the extreme score for each day in the full dataset, based on the function above, for each of the absolute returns, highest returns, and lowest returns. this is based on the gpd survival function
df['abs_extreme_score'] = daily_extreme_score(df['abs_RET'].fillna(0), abs_fit, tail='upper')
df['upper_extreme_score'] = daily_extreme_score(df['RET'].fillna(0), upper_fit, tail='upper')
df['lower_extreme_score'] = daily_extreme_score((-df['RET']).fillna(0), lower_fit, tail='upper')

# calculate the number of absolute returns above our threshold
df['abs_exceed'] = (df['abs_RET'] > abs_fit['threshold']).astype(int)
df['upper_exceed'] = (df['RET'] > upper_fit['threshold']).astype(int)
df['lower_exceed'] = ((-df['RET']) > lower_fit['threshold']).astype(int)

# -------------------------------
# QUARTERLY AGGREGATION (QE)
# -------------------------------
# take all of the columns we measured and aggregate them to the quarterly level. that is what the resample('QE') function does
df_q = df.set_index('date').resample('QE').agg({
    'RET': ['mean', 'std'],
    'abs_RET': ['mean', 'max'],
    'abs_exceed': 'sum',
    'upper_exceed': 'sum',
    'lower_exceed': 'sum',
    'abs_extreme_score': 'mean',
    'upper_extreme_score': 'mean',
    'lower_extreme_score': 'mean'
})
df_q.columns = ['_'.join(col).strip() for col in df_q.columns.values]
df_q = df_q.reset_index().rename(columns={'date': 'quarter_end'})

days_per_q = df.set_index('date').resample('QE').size().values
df_q['days_in_quarter'] = days_per_q

# calculate the fraction of the days in the quarter that exceeded the GPD threshold
df_q['abs_exceed_ratio'] = df_q['abs_exceed_sum'] / df_q['days_in_quarter']
df_q['upper_exceed_ratio'] = df_q['upper_exceed_sum'] / df_q['days_in_quarter']
df_q['lower_exceed_ratio'] = df_q['lower_exceed_sum'] / df_q['days_in_quarter']

# -------------------------------
# CALCULATE QUARTERLY EVT METRICS (INCLUDING VAR/ES)
# -------------------------------
abs_mean_excess, abs_max_excess, abs_es_q = [], [], []
upper_mean_excess, upper_max_excess, upper_es_q = [], [], []
lower_mean_excess, lower_max_excess, lower_es_q = [], [], []

for q, group in df.set_index('date').resample('QE'):
    # ABSOLUTE RETURNS
    exceed = group['abs_RET'] - abs_fit['threshold']
    exceed = exceed[exceed > 0]
    abs_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    abs_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    xi, sigma, threshold = abs_fit['xi'], abs_fit['sigma'], abs_fit['threshold']
    var_q = threshold + abs_fit['gpd_dist'].ppf(0.99)
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    abs_es_q.append(es_q)

    # UPPER RETURNS
    exceed = group['RET'] - upper_fit['threshold']
    exceed = exceed[exceed > 0]
    upper_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    upper_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    xi, sigma, threshold = upper_fit['xi'], upper_fit['sigma'], upper_fit['threshold']
    var_q = threshold + upper_fit['gpd_dist'].ppf(0.99)
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    upper_es_q.append(es_q)

    # LOWER RETURNS
    exceed = (-group['RET']) - lower_fit['threshold']
    exceed = exceed[exceed > 0]
    lower_mean_excess.append(exceed.mean() if len(exceed) > 0 else 0.0)
    lower_max_excess.append(exceed.max() if len(exceed) > 0 else 0.0)
    xi, sigma, threshold = lower_fit['xi'], lower_fit['sigma'], lower_fit['threshold']
    var_q = threshold + lower_fit['gpd_dist'].ppf(0.99)
    es_q = (var_q + (sigma - xi*(var_q - threshold)) / (1 - xi)) if xi < 1 else np.inf
    lower_es_q.append(es_q)

df_q['abs_mean_excess'] = abs_mean_excess
df_q['abs_max_excess'] = abs_max_excess
df_q['abs_es_q'] = abs_es_q

df_q['upper_mean_excess'] = upper_mean_excess
df_q['upper_max_excess'] = upper_max_excess
df_q['upper_es_q'] = upper_es_q

df_q['lower_mean_excess'] = lower_mean_excess
df_q['lower_max_excess'] = lower_max_excess
df_q['lower_es_q'] = lower_es_q

# -------------------------------
# TARGET VARIABLE
# -------------------------------
df_q['municipal_decline'] = (df_q['quarter_end'] >= pd.to_datetime('2013-07-01')).astype(int)

# -------------------------------
# FEATURES
# -------------------------------
evt_features = [
    'abs_mean_excess', 'abs_max_excess', 'abs_es_q',
    'upper_mean_excess', 'upper_max_excess', 'upper_es_q',
    'lower_mean_excess', 'lower_max_excess', 'lower_es_q',
    'abs_exceed_ratio', 'upper_exceed_ratio', 'lower_exceed_ratio',
    'abs_extreme_score_mean', 'upper_extreme_score_mean', 'lower_extreme_score_mean',
    'abs_RET_max', 'abs_RET_mean'
]
stock_features = ['RET_std', 'RET_mean']
features = evt_features + stock_features

df_q[features] = df_q[features].fillna(0.0)

# -------------------------------
# MODEL PREPARATION
# -------------------------------
# prepare the data for modeling:
# X: the input features for each quarter (EVT, stock, and economic variables)
# y: the target variable (whether municipal decline occurs that quarter)
# indices: just the row numbers, used to track which data points go where
# then split the data into training and test sets:
# X_train, y_train: features and target for training the model
# X_test, y_test: features and target for testing the model
# idx_train, idx_test: the original row indices corresponding to the training and test sets
# the split is 70% training, 30% test, keeping a similar balance of the target variable in both sets
X = df_q[features].values
y = df_q['municipal_decline'].values
indices = np.arange(len(df_q))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# standardize features so they’re on the same scale for the model
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# train logistic regression, using class_weight='balanced' so rare events (municipal decline) count more
# y_pred: predicted classes, y_prob: predicted probabilities for decline
clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
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

df_q['pred_prob'] = np.nan
df_q.loc[idx_test, 'pred_prob'] = y_prob
df_q.loc[idx_train, 'pred_prob'] = clf.predict_proba(X_train_s)[:,1]

df_q.to_csv("updated_gpd_for_each_quarter.csv", index=False)
print("Saved quarterly features + preds to 'updated_gpd_for_each_quarter.csv'")

decline_threshold = 0.5  

predicted_decline_dates = df_q.loc[
    (df_q['pred_prob'] >= decline_threshold),
    'quarter_end'
].apply(lambda x: x.date()).tolist()

actual_decline_dates = df_q.loc[
    (df_q['municipal_decline'] == 1),
    'quarter_end'
].apply(lambda x: x.date()).tolist()

overlap_decline_dates = df_q.loc[
    (df_q['municipal_decline'] == 1) & (df_q['pred_prob'] >= decline_threshold),
    'quarter_end'
].apply(lambda x: x.date()).tolist()


print("\n--- Municipal Decline Date Analysis (Full Dataset) ---")
print(f"Total Quarters in Dataset: {len(df_q)}")
print(f"Total Actual Decline Quarters (Target=1): {len(actual_decline_dates)}")
print(f"Total Predicted Decline Quarters (Prob >= {decline_threshold}): {len(predicted_decline_dates)}")

print("\n**Dates Predicted to Have Municipal Decline:**")
for date in predicted_decline_dates:
    print(date)

print("\n**Dates That Actually Had Municipal Decline (Target = 1):**")
for date in actual_decline_dates:
    print(date)

print("\n**Dates of Overlap (True Positives):**")
for date in overlap_decline_dates:
    print(date)

tp = len(overlap_decline_dates)
fp = len(predicted_decline_dates) - tp
fn = len(actual_decline_dates) - tp

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")