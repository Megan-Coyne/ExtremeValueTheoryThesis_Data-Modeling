import pandas as pd
from pyextremes import EVA
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/ECONOMIC_CONDITIONS_INDEX_DETROIT.csv", 
    parse_dates=["observation_date"],
    index_col="observation_date"
)

series = df["DWLAGRIDX"].astype(float).dropna()

series = series.loc["1990-01-01":]

series = series - (series.index - pd.to_datetime("1992-01-01")) / pd.to_timedelta("365.2425D") * 2.87e-3

model = EVA(series)

# get the yearly minima (block size ~ 1 year)
model.get_extremes(
    method="BM",
    block_size="365.2425D",
    extremes_type="low",  # for minima
    errors="ignore"     
)

model.plot_extremes()
plt.title("Block Minima of Detroit Economic Conditions Index")
plt.show()
