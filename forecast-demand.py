import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

def clean_and_save():
    filename = "HistoricalProductDemand-kaggle-felixzhao-productdemandforecasting.csv"
    dirname = "DATA"
    path = os.path.join(dirname, filename)
    df = pd.read_csv(path, parse_dates=['Date'])

    df = (
        pd.concat(
            [
            df[['Product_Code', 'Warehouse', 'Product_Category']].astype('category'),df[['Date', 'Order_Demand']]
            ],
            axis='columns'
        )
    )

    # There are 2160 different Product Codes
    # There are 4 different Warehouses
    # There are 33 different product categories
    # for col in df.columns[:3]:
    #     print(f"\n\n{df[col].unique()}")

    df = df.dropna().set_index('Date')
    df.to_pickle("forecast_demand.pkl")

if __name__ == '__main__':
    from neuralprophet import NeuralProphet
    new = pd.read_pickle("forecast_demand.pkl")
    print(new.head())
    dir(NeuralProphet)
