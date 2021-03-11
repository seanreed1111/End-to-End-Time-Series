import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import os
from neuralprophet import NeuralProphet
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

filename = "HistoricalProductDemand-kaggle-felixzhao-productdemandforecasting.csv"
dirname = "DATA"
path = os.path.join(dirname, filename)

df = pd.read_csv(path, parse_dates=['Date'])
# print(df.info())
cats = df[['Product_Code', 'Warehouse', 'Product_Category']].astype('category')
# print(cats.info())
df = pd.concat([cats, df[['Date', 'Order_Demand']]], axis='columns')
print(df.info())