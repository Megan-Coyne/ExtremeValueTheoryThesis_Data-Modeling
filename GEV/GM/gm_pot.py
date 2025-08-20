import pandas as pd
from pyextremes import EVA
import matplotlib.pyplot as plt


df = pd.read_csv(
    "data/GMStock.csv",
    parse_dates=["date"],
    index_col="date"
)

series = df["RETX"].astype(float).dropna()

series = series.loc["1990-01-01":]

series = series - (series.index - pd.to_datetime("1992-01-01")) / pd.to_timedelta("365.2425D") * 2.87e-3

model = EVA(series)

model.get_extremes(method="BM", block_size="365.2425D", extremes_type="low")

model.plot_extremes()
plt.title("Block Minima of General Motors Monthly Stock Returns")
plt.show()  
