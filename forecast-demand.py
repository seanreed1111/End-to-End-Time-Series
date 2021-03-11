import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import os
import math

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

    df = df.dropna().set_index('Date').sort_index()
    
    def gt0(x):
        if x> 0:
            return x
        else:
            return 0
    
    df.loc[:,'Order_Demand'] = df['Order_Demand'].apply(gt0)
    df.to_pickle("DATA/forecast-demand.pkl")
    

if __name__ == '__main__':
    #from neuralprophet import NeuralProphet
    # clean_and_save()
    df = pd.read_pickle("DATA/forecast-demand.pkl")

    products = df.groupby('Product_Code')
    # test one product for forecasting
    Product_2171 = products.get_group("Product_2171").reset_index()
    Product_2171.plot(x='Date', y='Order_Demand', ylim = (-20,80))
    plt.show()
