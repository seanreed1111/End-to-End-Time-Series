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
# cats = df[['Product_Code', 'Warehouse', 'Product_Category']].astype('category')
# df = pd.concat([cats, df[['Date', 'Order_Demand']]], axis='columns')


df = (
    pd.concat(
        [
        df[['Product_Code', 'Warehouse', 'Product_Category']].astype('category'),df[['Date', 'Order_Demand']]
        ],
        axis='columns')
)
#print(df.info())
# There are 2160 different Product Codes
# There are 4 different Warehouses
# There are 33 different product categories
for col in df.columns[:3]:
    print(f"\n\n{df[col].unique()}")
    