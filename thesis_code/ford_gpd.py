from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from GPD.generalized_pareto import fit_gpd, plot_gpd_fit
from GoodnessOfFit.KStest import ks_test, plot_ks_test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# Define metrics and which tail to consider
metric_tails = {
    'RET': 'both',      # large positive or negative returns
    'PRC': 'lower',     # very low prices may indicate distress
    'VOL': 'upper',     # unusually high volume
    'BIDLO': 'lower',   # extremely low bids may indicate stress
    'ASKHI': 'upper',   # unusually high asks
}

def fit_all_metrics(df, company_name, quantile=0.95):
    """Fit GPD and run KS test for all relevant metrics."""
    results = {}

    for metric, tail in metric_tails.items():
        if metric not in df.columns:
            continue  # skip missing columns

        data = df[metric].dropna()

        # Handle absolute values for symmetric tails
        if tail == 'both':
            data = data.abs()

        # Fit GPD
        fit_result = fit_gpd(data, f"{company_name} - {metric}", quantile=quantile)
        results[metric] = fit_result

        print(f"--- {company_name} - {metric} ---")
        print("Shape ξ:", fit_result["xi"])
        print("Scale σ:", fit_result["sigma"])
        print("Threshold:", fit_result["threshold"])

        # Plot PDF fit
        plot_gpd_fit(data, fit_result)

        # KS Test
        transform = lambda x: x[x > fit_result["threshold"]] - fit_result["threshold"]
        ks_result = ks_test(data, dist=fit_result["gpd_dist"], transform=transform)
        plot_ks_test(ks_result, title=f"{company_name} - {metric} GPD KS Test")

    return results


df_ford = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")
df_ford['date'] = pd.to_datetime(df_ford['date'])

ford_results = fit_all_metrics(df_ford, "Ford")


def daily_extreme_score(data, fit_result, tail='upper'):
    threshold = fit_result["threshold"]
    gpd_dist = fit_result["gpd_dist"]
    data = np.asarray(data)

    if tail == 'upper':
        exceedances = data - threshold
        exceedances[exceedances < 0] = 0
        scores = gpd_dist.sf(exceedances)  # 1 - CDF
    elif tail == 'lower':
        exceedances = threshold - data
        exceedances[exceedances < 0] = 0
        scores = gpd_dist.cdf(exceedances)
    elif tail == 'both':
        exceedances = np.abs(data - threshold)
        scores = gpd_dist.sf(exceedances)
    else:
        raise ValueError("tail must be 'upper', 'lower', or 'both'")

    return scores


def aggregate_metrics_to_company(df, fit_results, metric_tails, method='mean', weights=None):
    metric_scores = {}
    for metric, fit in fit_results.items():
        tail = metric_tails.get(metric, 'upper')
        metric_scores[metric] = daily_extreme_score(df[metric].fillna(0), fit, tail=tail)

    all_scores = np.vstack(list(metric_scores.values()))  # metrics x days

    if method == 'mean':
        return np.mean(all_scores, axis=0)
    elif method == 'max':
        return np.max(all_scores, axis=0)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Provide weights for weighted aggregation")
        weighted_scores = np.zeros(all_scores.shape[1])
        for i, metric in enumerate(metric_scores.keys()):
            weighted_scores += all_scores[i] * weights.get(metric, 1)
        return weighted_scores / sum(weights.values())
    else:
        raise ValueError("method must be 'mean', 'max', or 'weighted'")


def aggregate_companies_to_city(company_scores, method='mean', weights=None):
    all_scores = np.vstack(list(company_scores.values()))  # companies x days

    if method == 'mean':
        return np.mean(all_scores, axis=0)
    elif method == 'max':
        return np.max(all_scores, axis=0)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Provide weights for weighted aggregation")
        weighted_scores = np.zeros(all_scores.shape[1])
        for i, company in enumerate(company_scores.keys()):
            weighted_scores += all_scores[i] * weights.get(company, 1)
        return weighted_scores / sum(weights.values())
    else:
        raise ValueError("method must be 'mean', 'max', or 'weighted'")


ford_company_score = aggregate_metrics_to_company(df_ford, ford_results, metric_tails, method='mean')
# If you have GM, Chrysler, etc., do the same and store in a dict:
company_scores = {
    'Ford': ford_company_score,
    # 'GM': gm_company_score,
    # 'Chrysler': chrysler_company_score,
}

city_score = aggregate_companies_to_city(company_scores, method='mean')

dates = df_ford['date']  # assuming all company data is aligned by date
city_score_df = pd.DataFrame({
    'date': dates,
    'city_extreme_score': city_score
})

# Example: Detroit bankruptcy July 18, 2013
city_score_df['municipal_decline'] = 0
city_score_df.loc[city_score_df['date'] >= '2013-07-18', 'municipal_decline'] = 1


X = city_score_df[['city_extreme_score']].values
y = city_score_df['municipal_decline'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


city_score_df['predicted_prob'] = model.predict_proba(X)[:, 1]

plt.figure(figsize=(12, 5))
plt.plot(city_score_df['date'], city_score_df['predicted_prob'], label='Predicted Decline Probability')
plt.axvline(pd.to_datetime('2013-07-18'), color='red', linestyle='--', label='Bankruptcy Date')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.title('City-Level Predicted Decline Probability Over Time')
plt.legend()
plt.show()