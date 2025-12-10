import pandas as pd
import numpy as np
from datetime import datetime

from GPD.generalized_pareto import fit_gpd, plot_gpd_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def daily_extreme_score(data, fit_result, tail='upper'):
    threshold 