import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_ford = pd.read_csv("/Users/megancoyne/school/thesis/thesis_code/data/FordStock_2015.csv")
return_data = df_ford['RET']

print("First few returns:")
print(return_data.head())

return_data = np.sort(return_data)
cdf = np.arange(1, len(return_data) + 1) / len(return_data)

plt.step(return_data, cdf)
plt.xlabel("Return")
plt.ylabel("Empirical CDF")
plt.title("Empirical CDF of Ford Returns")
plt.show()
