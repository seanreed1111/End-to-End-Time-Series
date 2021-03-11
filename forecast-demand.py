import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    #ensure minimum forecast is zero
    # def gt0(x):
    #     if x > 0:
    #         return x
    #     else:
    #         return 0
    
    #df.loc[:,'Order_Demand'] = df['Order_Demand'].apply(gt0)
    df.loc[:,'Order_Demand'] = df['Order_Demand'].apply(lambda x: 0 if x < 0 else x)
    df.to_pickle("DATA/forecast-demand.pkl")
    

if __name__ == '__main__':
    #from neuralprophet import NeuralProphet
    # clean_and_save()
    df = pd.read_pickle("DATA/forecast-demand.pkl")

    products = df.groupby('Product_Code')
    
    # prodmed, prodmax = products.median(),products.max()
    # print(prodmin.head(10))
    
    # _ = sns.boxplot(y=prodmed['Order_Demand'])
    # _ = sns.boxplot(y=prodmax['Order_Demand'])
    # plt.show()
    #describe.reset_index().plot(x='Date')
    # test one product for forecasting
    
    Product_2001 = products.get_group("Product_2001").reset_index()
    Product_2001.plot(x='Date', y='Order_Demand')
    plt.show()
    print(Product_2001.info())
