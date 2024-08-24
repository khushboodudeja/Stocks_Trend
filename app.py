import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import yfinance as yf  # type: ignore
import streamlit as st  # type: ignore
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore

# Set the start and end date for data retrieval
start = '2010-1-1'
end = '2019-12-31'

# Streamlit title
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download stock data
if user_input:
    df = yf.download(user_input, start=start, end=end, progress=False)
else:
    st.warning("Please enter a valid stock ticker.")

# Check if data is available
if df.empty:
    st.write("No data available for the selected ticker.")
    st.stop()  # Stop execution if no data

# Describe data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualization: Closing Price vs Time
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title('Closing Price vs Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Visualization: Moving Averages
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100-Day Moving Average')
plt.plot(ma200, 'g', label='200-Day Moving Average')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.title('Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the model with error handling
try:
    model = load_model('keras_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

# Prepare x_test and y_test arrays
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

# Convert to numpy arrays and reshape
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Ensure shape is (samples, timesteps, features)

# Make predictions
try:
    y_predicted = model.predict(x_test)
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.stop()

# 
