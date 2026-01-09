# ============================================================
# EVT-Based Early Warning Model (Leakage-Safe, Reduced, Valid)
# ============================================================

import pandas as pd
import numpy as np

from GPD.generalized_pareto import fit_gpd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD AND PREPARE DAILY DATA
# ============================================================

df = pd.read_csv(
    "/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv"
)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Daily transformations
df['abs_RET'] = df['RET'].abs()
df['log_volume'] = np.log1p(df['VOL'])
df['rel_spread'] = (df['ASK'] - df['BID']) / ((df['ASK'] + df['BID']) / 2)

# ============================================================
# 2. DEFINE TRAINING WINDOW (STRICT TIME SPLIT)
# ============================================================

train_end = pd.to_datetime('2012-12-31')

df_train = df[df['date'] <= train_end].copy()
df_test  = df[df['date'] > train_end].copy()

# ============================================================
# 3. FIT GPD ON TRAINING DATA ONLY (NO LEAKAGE)
# ============================================================

quantile = 0.95

abs_fit = fit_gpd(
    df_train['abs_RET'].dropna(),
    "abs_RET (train)",
    quantile=quantile
)

upper_fit = fit_gpd(
    df_train['RET'].dropna(),
    "RET upper tail (train)",
    quantile=quantile
)

lower_fit = fit_gpd(
    (-df_train['RET']).dropna(),
    "RET lower tail (train)",
    quantile=quantile
)

# ============================================================
# 4. QUARTERLY EVT FEATURE CONSTRUCTION
#    (APPLY TRAIN-FITTED GPD TO ALL DATA)
# ============================================================

def compute_quarterly_features(df, abs_fit, upper_fit, lower_fit):
    rows = []

    for q, group in df.set_index('date').resample('QE'):
        if len(group) == 0:
            continue

        row = {'quarter_end': q}
        n = len(group)

        # --- EVT: absolute returns ---
        exc_abs = group['abs_RET'] - abs_fit['threshold']
        exc_abs = exc_abs[exc_abs > 0]

        row['abs_exceed_ratio'] = len(exc_abs) / n
        row['abs_mean_excess'] = exc_abs.mean() if len(exc_abs) > 0 else 0.0

        # --- EVT: upper tail ---
        exc_up = group['RET'] - upper_fit['threshold']
        exc_up = exc_up[exc_up > 0]

        row['upper_exceed_ratio'] = len(exc_up) / n

        # --- EVT: lower tail ---
        exc_low = (-group['RET']) - lower_fit['threshold']
        exc_low = exc_low[exc_low > 0]

        row['lower_exceed_ratio'] = len(exc_low) / n

        # --- Controls ---
        row['RET_mean'] = group['RET'].mean()
        row['RET_std'] = group['RET'].std()
        row['log_volume_mean'] = group['log_volume'].mean()
        row['rel_spread_mean'] = group['rel_spread'].mean()

        rows.append(row)

    return pd.DataFrame(rows)

df_q = compute_quarterly_features(df, abs_fit, upper_fit, lower_fit)

# ============================================================
# 5. TARGET VARIABLE (KNOWN DECLINE PERIOD)
# ============================================================

df_q['municipal_decline'] = (
    df_q['quarter_end'] >= pd.to_datetime('2013-07-01')
).astype(int)

# ============================================================
# 6. FINAL FEATURE SET (REDUCED + DEFENSIBLE)
# ============================================================

features = [
    'abs_exceed_ratio',
    'upper_exceed_ratio',
    'lower_exceed_ratio',
    'abs_mean_excess',
    'RET_mean',
    'RET_std',
    'log_volume_mean',
    'rel_spread_mean'
]

df_q[features] = df_q[features].fillna(0.0)

# ============================================================
# 7. TIME-BASED TRAIN / TEST SPLIT (QUARTERS)
# ============================================================

train_q = df_q[df_q['quarter_end'] <= train_end].copy()
test_q  = df_q[df_q['quarter_end'] > train_end].copy()

X_train = train_q[features].values
y_train = train_q['municipal_decline'].values

X_test = test_q[features].values
y_test = test_q['municipal_decline'].values

# ============================================================
# 8. MULTICOLLINEARITY CHECK (TRAINING ONLY)
# ============================================================

corr = pd.DataFrame(
    X_train, columns=features
).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Predictor Correlation Matrix (Training Data)")
plt.tight_layout()
plt.show()

# ============================================================
# 9. STANDARDIZATION (TRAIN ONLY)
# ============================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 10. LOGISTIC REGRESSION (LOW-DIMENSIONAL)
# ============================================================

clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs'
)

clf.fit(X_train_s, y_train)

# ============================================================
# 11. EVALUATION (POST-2013 ONLY)
# ============================================================

y_prob = clf.predict_proba(X_test_s)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n--- TEST PERFORMANCE (POST-2013) ---")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n--- CONFUSION MATRIX COUNTS ---")
print(f"True Positives  (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives  (TN): {tn}")

# ============================================================
# 12. QUARTER-LEVEL ERROR INTERPRETATION
# ============================================================

test_q = test_q.copy()
test_q['pred_prob'] = y_prob
test_q['y_pred'] = y_pred

print("\n--- TRUE POSITIVE QUARTERS ---")
for d in test_q[(test_q['municipal_decline'] == 1) & (test_q['y_pred'] == 1)]['quarter_end']:
    print(d.date())

print("\n--- FALSE POSITIVE QUARTERS ---")
for d in test_q[(test_q['municipal_decline'] == 0) & (test_q['y_pred'] == 1)]['quarter_end']:
    print(d.date())

print("\n--- FALSE NEGATIVE QUARTERS ---")
for d in test_q[(test_q['municipal_decline'] == 1) & (test_q['y_pred'] == 0)]['quarter_end']:
    print(d.date())

# ============================================================
# 13. COEFFICIENTS (INTERPRETABLE)
# ============================================================

coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': clf.coef_[0]
}).sort_values(by='coefficient', key=np.abs, ascending=False)

print("\n--- MODEL COEFFICIENTS ---")
print(coef_df)

# ============================================================
# 14. SAVE RESULTS
# ============================================================

test_q.to_csv("evt_logistic_results_corrected.csv", index=False)
print("\nSaved results to evt_logistic_results_corrected.csv")
