import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from neuralprophet import NeuralProphet
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

def convert_to_date(x):
	return datetime.strptime(x, '%Y %m %d %H')

aq_df_sh = pd.read_csv('~/DEVELOPMENT/DSDE/NP/RESOURCES/DATA/PRSA_Data_Shunyi_20130301-20170228.csv', parse_dates = [['year', 'month', 'day', 'hour']],date_parser=convert_to_date)

aq_df_wa = pd.read_csv('~/DEVELOPMENT/DSDE/NP/RESOURCES/DATA/PRSA_Data_Wanliu_20130301-20170228.csv', parse_dates = [['year', 'month', 'day', 'hour']],date_parser=convert_to_date)

aq_df_gu = pd.read_csv('~/DEVELOPMENT/DSDE/NP/RESOURCES/DATA/PRSA_Data_Gucheng_20130301-20170228.csv', parse_dates = [['year', 'month', 'day', 'hour']],date_parser=convert_to_date)

aq_df = pd.concat([aq_df_sh, aq_df_wa,aq_df_gu], ignore_index=True, sort=False)
aq_df=aq_df.drop(['No'], axis=1)

aq_df_daily = aq_df.set_index('year_month_day_hour').groupby('station').resample('D').mean().reset_index()

#aq_df_daily.query("station=='Shunyi'")[['O3','TEMP']].plot()

#aq_df_daily.set_index('year_month_day_hour').groupby('station')[['O3','TEMP']].plot()
#plt.show()

#aq_df_final = aq_df_daily[['year_month_day_hour','O3','TEMP','station']].rename({'year_month_day_hour':'ds','O3':'y'}, axis='columns')
aq_df_final = aq_df_daily[['year_month_day_hour','O3','station']].rename({'year_month_day_hour':'ds','O3':'y'}, axis='columns')
#print(aq_df_final.head())
shunyi = aq_df_final.query('station == "Shunyi"')[["ds",'y']]
print(shunyi.head())

# stations = aq_df_final.groupby('station')
# print(stations.head())
target = pd.DataFrame()

# for station in stations.groups:
#     group = stations.get_group(station)
#     print(station)
m = NeuralProphet(
    n_forecasts=366,
    n_lags=2,
    changepoints_range=0.85,
    n_changepoints=30,
    epochs=10,
)
m.fit(shunyi, freq='D')
    
future = m.make_future_dataframe(shunyi, periods=366)
forecast = m.predict(future)  
m.plot(forecast)  
forecast = forecast.rename(columns={'yhat': 'yhat_'+'shunyi'})
target = pd.merge(target, forecast.set_index('ds'), how='outer', left_index=True, right_index=True)
plt.show()