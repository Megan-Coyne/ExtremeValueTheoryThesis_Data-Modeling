import pandas as pd
import numpy as np
from datetime import datetime

from GPD.generalized_pareto import fit_gpd, plot_gpd_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import norm


def daily_extreme_score(data, fit_result, tail='upper'):
    threshold = fit_result["threshold"]
    gpd_dist = fit_result["gpd_dist"]
    data = np.asarray(data)
    scores = np.zeros_like(data, dtype=float)

    if tail == 'upper':
        
        exceedances = data - threshold
        exceedances[exceedances < 0] = 0
        # given that a return exceeds the threshold, just how extreme was it compared to other extreme days 
        scores = np.where(exceedances > 0, gpd_dist.sf(exceedances), 0.0)
    else:
        raise ValueError("tail must be 'upper', 'lower', or 'both'")
    
    return scores

# calculates the Value-at-Risk and Expected Shortfall from the gpd fit at a given probability level
def VaR_ES_from_gpd(fit_result, prob):
    threshold = fit_result["threshold"]
    xi = fit_result["xi"]       # get shape parameter
    sigma = fit_result["sigma"] # get scale parameter
    gpd_dist = fit_result["gpd_dist"]
    
    var = threshold + gpd_dist.ppf(1 - prob)
    
    if xi < 1:
        es = (var + (sigma - xi * (var - threshold)) / (1 - xi))
    else:
        es = np.inf
    
    return var, es

# read in the dataset, extract the data based on date, compute the absolute returns
df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['abs_RET'] = df['RET'].abs()

# read in the economic conditions index dataset, extract the data based on date, merge this data with the stock data as an additional column in the dataset
econ_df = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv")
econ_df['date'] = pd.to_datetime(econ_df['observation_date'])
df = df.merge(econ_df, on='date', how='left')
df['econ_index'] = df['DWLAGRIDX'].fillna(method='ffill').fillna(0.0)

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

# calculate the extreme score for each day in the full dataset, based on the function above, for each of the absolute returns, highest returns, and lowest returns. this is based on the gpd survival function
df['abs_extreme_score'] = daily_extreme_score(df['abs_RET'].fillna(0), abs_fit, tail='upper')
df['upper_extreme_score'] = daily_extreme_score(df['RET'].fillna(0), upper_fit, tail='upper') 
df['lower_extreme_score'] = daily_extreme_score((-df['RET']).fillna(0), lower_fit, tail='upper')

# calculate the number of absolute returns above our threshold
df['abs_exceed'] = (df['abs_RET'] > abs_fit['threshold']).astype(int)
df['upper_exceed'] = (df['RET'] > upper_fit['threshold']).astype(int) 
df['lower_exceed'] = ((-df['RET']) > lower_fit['threshold']).astype(int)

# take all of the columns we measured and aggregate them to the quarterly level. that is what the resample('Q') function does
df_q = df.set_index('date').resample('Q').agg({
    'RET': ['mean', 'std'],
    'abs_RET': ['mean', 'max'],
    'abs_exceed': 'sum',
    'upper_exceed': 'sum',
    'lower_exceed': 'sum',
    'abs_extreme_score': 'mean',
    'upper_extreme_score': 'mean',
    'lower_extreme_score': 'mean',
    'econ_index': 'mean'
})

df_q.columns = ['_'.join(col).strip() for col in df_q.columns.values]
df_q = df_q.reset_index().rename(columns={'date': 'quarter_end'})

days_per_q = df.set_index('date').resample('Q').size().values
df_q['days_in_quarter'] = days_per_q


df_q['abs_exceed_ratio'] = df_q['abs_exceed_sum'] / df_q['days_in_quarter']
df_q['upper_exceed_ratio'] = df_q['upper_exceed_sum'] / df_q['days_in_quarter']
df_q['lower_exceed_ratio'] = df_q['lower_exceed_sum'] / df_q['days_in_quarter']

# calculate the VaR and ES at the 99% level for each of the three gpd fits
# interpretation of VaR: given my fitted GPD for exceedances above the 95% quantile, VaR is the return level expected to be exceeded only 1% of the time
# interpretation of ES: given the same GPD, ES is the average return when the 1% tail event occurs, that is, the expected size of losses beyond the VaR threshold
df_q['abs_var_99'], df_q['abs_es_99'] = VaR_ES_from_gpd(abs_fit, 0.01)
df_q['upper_var_99'], df_q['upper_es_99'] = VaR_ES_from_gpd(upper_fit, 0.01)
df_q['lower_var_99'], df_q['lower_es_99'] = VaR_ES_from_gpd(lower_fit, 0.01)

# renaming all of the columns in our aggregated quarterly dataframe
df_q = df_q.rename(columns={
    'RET_mean': 'quarter_ret_mean',
    'RET_std': 'quarter_ret_std',
    'abs_RET_mean': 'quarter_absret_mean',
    'abs_RET_max': 'quarter_absret_max',
    'abs_extreme_score_mean': 'quarter_abs_score_mean',
    'upper_extreme_score_mean': 'quarter_upper_score_mean',
    'lower_extreme_score_mean': 'quarter_lower_score_mean',
    'econ_index_mean': 'econ_index_mean',
    'abs_var_99': 'abs_var_es',
    'upper_var_99': 'upper_var_es',
    'lower_var_99': 'lower_var_es',
    # need to add the VaR and ES
    # leave out all the predictors and just do the VaR and ES
})

# define our target variable: quarters after Q2 2013 (the time of Detroit's municipal bond downgrade) are labeled as 1, prior quarters as 0
df_q['municipal_decline'] = (df_q['quarter_end'] >= pd.to_datetime('2013-07-01')).astype(int)

# all of our predictor values, including those calculated from EVT, the stock returns, and the Detroit economic conditions index
evt_features = [
    'abs_exceed_ratio',
    'upper_exceed_ratio',
    'lower_exceed_ratio',
    'quarter_abs_score_mean',
    'quarter_upper_score_mean',
    'quarter_lower_score_mean',
    'quarter_absret_max',
    'quarter_absret_mean',
    'abs_var_es',
    'upper_var_es',
    'lower_var_es'
]
stock_features = ['quarter_ret_std', 'quarter_ret_mean']
econ_features = ['econ_index_mean']
features = evt_features + stock_features + econ_features

# fill any NaN values with 0.0 for our modeling
df_q[features] = df_q[features].fillna(0.0)

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
clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs') # need to understand balanced
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:,1]

print("Test set classification report:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC (test):", roc_auc_score(y_test, y_prob))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_pred))

# show which features matter most, store predicted probabilities, and save everything to a file
coef = clf.coef_[0]
for f, c in sorted(zip(features, coef), key=lambda x: -abs(x[1])):
    print(f"{f:30s} coef={c:.4f}")

df_q['pred_prob'] = np.nan
df_q.loc[idx_test, 'pred_prob'] = y_prob
df_q.loc[idx_train, 'pred_prob'] = clf.predict_proba(X_train_s)[:,1]

df_q.to_csv("ford_quarterly_evt_features_and_preds.csv", index=False)
print("Saved quarterly features + preds to 'ford_quarterly_evt_features_and_preds.csv'")
