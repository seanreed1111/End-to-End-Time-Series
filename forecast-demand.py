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
    from neuralprophet import NeuralProphet, set_random_seed
    # clean_and_save()
    set_random_seed(42)
    df = pd.read_pickle("DATA/forecast-demand.pkl")
    # df.loc[:,'Order_Demand'] = df['Order_Demand'].apply(lambda x: math.log(x+1))
    products = df.groupby('Product_Code')
    
    # prodmed, prodmax = products.median(),products.max()
    # print(prodmin.head(10))
    
    # _ = sns.boxplot(y=prodmed['Order_Demand'])
    # _ = sns.boxplot(y=prodmax['Order_Demand'])
    # plt.show()
    #describe.reset_index().plot(x='Date')
    # test one product for forecasting
    
    Product_1766 = (products.resample('D')
                    .median()
                    .query("Product_Code == 'Product_1766'")
                    .reset_index()
                    )

    # Product_1766.plot(x='Date', y='Order_Demand')
    # plt.show()
    Product_1766 = (Product_1766[['Date','Order_Demand']]
                    .rename(columns={'Date':'ds', 'Order_Demand':'y'})
    )
    # print(Product_1766.describe())
    # print(Product_1766.tail(200))
    

    m = NeuralProphet(
        n_forecasts=60,
        n_lags=15,
        #seasonality_mode="multiplicative",
        epochs=10
    )


    m.fit(Product_1766, freq='D')

    future = m.make_future_dataframe(Product_1766, 
                                     periods=60,
                                     n_historic_predictions=len(Product_1766))
    forecast = m.predict(future)  
    m.plot(forecast)
    plt.show()
    
    # print(df.index)
    # print(Product_1766.index)
    
    # df_train, df_val = m.split_df(Product_1766, valid_p=0.2, freq='D')
    # train_metrics = m.fit(df_train, freq='D')
    # val_metrics = m.test(df_val)
    # print(train_metrics)
    # print(val_metrics)
    metrics = m.fit(Product_1766, validate_each_epoch=True, valid_p=0.2, freq='D')
    print(metrics)