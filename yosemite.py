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