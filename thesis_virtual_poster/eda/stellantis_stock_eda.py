from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# read datasdet using pandas
df = pd.read_csv('/Users/megancoyne/thesis/eda/Stellantis.csv')
df.head()

print(df)
print(df.shape)
print(df.info())
print(df.nunique())
print(df.describe())
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Close'] = le.fit_transform(df['Close'])
print(df['Close'])

# VISUALIZE

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.histplot(df['Close'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Histogram of Close')

sns.histplot(df['Open'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Histogram of Open')
plt.show()