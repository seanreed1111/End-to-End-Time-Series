import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import matplotlib.pyplot as plt
set_log_level("ERROR")
df = pd.read_csv("~/DEVELOPMENT/DSDE/NP/RESOURCES/DATA/yosemite_temps.csv",parse_dates=['ds'])
# print(df.info())
# print(df.head())

m = NeuralProphet(
    n_lags=12,
    changepoints_range=0.95,
    n_changepoints=30,
    weekly_seasonality=False,
    batch_size=64,
    epochs=10,
    learning_rate=1.0,
)
metrics = m.fit(df, freq='5min')


future = m.make_future_dataframe(df, n_historic_predictions=True)
forecast = m.predict(future)
fig = m.plot(forecast)

plt.show()

'''
Multi-step forecast
To predict multiple steps into the future, we could 'unroll' our single-step model, by predicting a step ahead, adding the forecasted value to our data, and then forecasting the next step until we reach the horizon we are interested in. However, there is a better way to do this: We can directly forecast multiple steps ahead with NeuralProphet.

We can set n_forecasts to the desired number of steps we would like to forecast (also known as 'forecast horizon'). NeuralProphet will forecast n_forecasts steps into the future, at every single step. Thus, we have n_forecasts overlapping predictions of vaying age at every historic point.

When icreasing the forecast horizon n_forecasts, we should also increase the number of past observations n_lags to at least the same value.

Here, we forecast the next 3 hours based on the last observed 6 hours, in 5-minute steps:
'''
m = NeuralProphet(
    n_lags=6*12,
    n_forecasts=3*12,
    changepoints_range=0.95,
    n_changepoints=30,
    weekly_seasonality=False,
    batch_size=64,
    epochs=10,    
    learning_rate=1.0,
)
metrics = m.fit(df, freq='5min')
future = m.make_future_dataframe(df, n_historic_predictions=True)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.show()

fig = m.plot(forecast[144:6*288])
plt.show()

# fig_comp = m.plot_components(forecast)
fig_param = m.plot_parameters()
plt.show()
'''
Larger forecast horizon¶
For predictions further into the future, you could reduce the resulution of the data. Using a 5-minute resolution may be useful for a high-resolution short-term forecast, but counter-productive for a long-term forecast. As we only have a limited amount of data (approx 2 months), we want to avoid over-specifying the model.

As an example: If we set the model to forecast 24 hours into the future (nforecasts=24*12) based on the last day's temperatures (n_lags=24*12), the number of parameters of our AR component grows to 24*12*24*12 = 82,944. However, we only have about 2*30*24*12 = 17,280 samples in our dataset. The model would be overspecified.

If we first downsample our data to hourly data, we reduce our dataset to 2*30*24=1440 and our model parameters to 24*24=576. Thus, we are able to fit the model. However, it would be better to collect more data.
'''

df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])
df_hourly = df.set_index('ds', drop=False).resample('H').mean().reset_index()

m = NeuralProphet(
    n_lags=24,
    n_forecasts=24,
    changepoints_range=0.95,
    n_changepoints=30,
    weekly_seasonality=False,
    learning_rate=0.3,
)
metrics = m.fit(df_hourly, freq='H')
future = m.make_future_dataframe(df_hourly, n_historic_predictions=True)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.show()

m = m.highlight_nth_step_ahead_of_each_forecast(24)
fig = m.plot_last_forecast(forecast, include_previous_forecasts=10)
plt.show()