import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

def sales_forecasting(data, product_id):

  # Filter data for selected product
  product_data = data[data['ProductID'] == product_id]
  
  # Train ARIMA model on historical sales data
  model = auto_arima(product_data['SalesAmount'], seasonal=False, m=12)
  
  # Make prediction for next 12 months 
  forecast = model.predict(n_periods=12)
  
  # Plot historical data and forecast
  plt.plot(product_data['Date'], product_data['SalesAmount'], label='Historical')
  plt.plot(forecast, label='Forecast')
  plt.title(f'Sales Forecast for Product {product_id}')
  plt.xlabel('Date')
  plt.ylabel('Sales Amount')
  plt.legend()
  
  plt.show()

  return forecast

data = pd.DataFrame({
  'Date': ['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01'],
  'ProductID': [1, 1, 1, 1],
  'SalesAmount': [100, 80, 90, 120]
})

product_forecast = sales_forecasting(data, 1)


