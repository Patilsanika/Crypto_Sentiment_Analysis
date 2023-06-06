import data_source.crypto_compare as cc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import logging
import os

# Initial ltsm code building off of

def import_tensorflow():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    
    tf.get_logger().setLevel(logging.ERROR)
    
    return tf
tf = import_tensorflow()  

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from data_source.crypto_compare import CryptoCompare

# class BtcLtsm:
#     def _init_(self, limit=1000):
#         self.limit = limit
#         self.cc = CryptoCompare()
#         self.scaler = MinMaxScaler()
#         self.model = None

#     def update_dataset(self):
#         df = self.cc.get_historical_price_hour('BTC', 'USD', limit=self.limit)
#         df = df[['time', 'close']]
#         df.to_csv('btc_price.csv', index=False)

#     def train(self):
#         df = pd.read_csv('btc_price_train.csv')
#         training_set = df.iloc[:, 1:2].values
#         training_set_scaled = self.scaler.fit_transform(training_set)

#         X_train = []
#         y_train = []
#         for i in range(60, len(training_set_scaled)):
#             X_train.append(training_set_scaled[i-60:i, 0])
#             y_train.append(training_set_scaled[i, 0])
#         X_train, y_train = np.array(X_train), np.array(y_train)

#         X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#         self.model = Sequential()
#         self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#         self.model.add(LSTM(units=50))
#         self.model.add(Dense(units=1))

#         self.model.compile(optimizer='adam', loss='mean_squared_error')
#         self.model.fit(X_train, y_train, epochs=10, batch_size=32)

#         self.model.save('btc_model.h5')

#     def load(self):
#         self.model = load_model('btc_model.h5')

#     def test_model(self):
#         df = pd.read_csv('btc_price_test.csv')
#         real_price = df.iloc[-1, 1]
#         test_set = df.iloc[:, 1:2].values
#         dataset_total = pd.concat((pd.read_csv('btc_price_train.csv').iloc[:, 1:2],
#                                    pd.read_csv('btc_price_test.csv').iloc[:, 1:2]), axis=0)
#         inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
#         inputs = inputs.reshape(-1,1)
#         inputs = self.scaler.transform(inputs)

#         X_test = []
#         for i in range(60, len(test_set)):
#             X_test.append(inputs[i-60:i, 0])
#         X_test = np.array(X_test)
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#         predicted_price = self.model.predict(X_test)
#         predicted_price = self.scaler.inverse_transform(predicted_price)

#         if predicted_price[-1] > real_price:
#             return "Buy"
#         elif predicted_price[-1] < real_price:
#             return "Sell"
#         else:
#             return "Hold"

class BtcLtsm:
    def __init__(self):
        self._data_source = cc.CryptoCompare()
        self._train_name_base = 'btc_price_train'
        self._test_name_base = 'btc_price_test'
        self._model_name_base = 'btc_ltsm'
        self._history = 60
        self._layer_size = 50
        self._dropout = 0.2
    
    def update_dataset(self, percent_train=0.98, limit=2000):
        try:
            ohlcv_df = self._data_source.get_daily_history('BTC', 'USDT', limit=limit)
            
            test_start_idx = int(len(ohlcv_df) * percent_train)
            
            train_df = ohlcv_df[:test_start_idx]
            test_df = ohlcv_df[test_start_idx:]
            train_df.to_csv(os.path.join('datasets', f'{self._train_name_base}.csv'), index=False)
            test_df.to_csv(os.path.join('datasets', f'{self._test_name_base}.csv'), index=False)
            return True
        except Exception as e:
            # Catch all exceptions and print the error message
            print(f"An error occurred: {e}")
            return False
    
    def train(self):
        train_file_name = os.path.join('datasets', f'{self._train_name_base}.csv')
        data_train = pd.read_csv(train_file_name)
        train_set = data_train.iloc[:, 1:2].values
        
        sc = MinMaxScaler(feature_range=(0, 1))
        train_set = sc.fit_transform(train_set)
        logging.debug(f'training set:\n{train_set}')
        
        # Creating a data structure with 60 timesteps and 1 output
        history = 60
        features_train = []
        results_train = []
        for i in range(history, len(train_set)):
            features_train.append(train_set[i-history:i, 0])
            results_train.append(train_set[i, 0])
        features_train, results_train = np.array(features_train), np.array(results_train)
        
        # Reshaping
        features_train = np.reshape(features_train, (features_train.shape[0], features_train.shape[1], 1))
        
        model_path = os.path.join('predictors/saved', f'{self._model_name_base}.h5')
        self._create_rnn(model_path, features_train, results_train)
    
    def load(self):
        model_path = os.path.join('predictors/saved', f'{self._model_name_base}.h5')
        self._regressor = load_model(model_path)
        
    def test_model(self):
        train_file_name = os.path.join('datasets', f'{self._train_name_base}.csv')
        test_file_name = os.path.join('datasets', f'{self._test_name_base}.csv')
        
        # Getting the real stock price of 2017
        dataset_test = pd.read_csv(test_file_name)
        real_stock_price = dataset_test.iloc[:, 1:2].values
        
        dataset_train = pd.read_csv(train_file_name)
        train_set = dataset_train.iloc[:, 1:2].values
        
        sc = MinMaxScaler(feature_range = (0, 1))
        train_set = sc.fit_transform(train_set)
        
        # Getting the predicted stock price of 2017
        dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        features_test = []
        for i in range(self._history, len(inputs)):
            features_test.append(inputs[i-self._history:i, 0])
        features_test = np.array(features_test)
        features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))
        predicted_stock_price = self._regressor.predict(features_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
        mape = mean_absolute_percentage_error(real_stock_price,predicted_stock_price)
        accuracy = 100 - mape

        print(f'RMSE: {rmse:.2f}')
        print(f'MAPE: {mape:.2f}%')
        print(f'Accuracy: {accuracy:.2f}%')

        if (predicted_stock_price[-1] > real_stock_price).any():
            print("Buy")
        elif (predicted_stock_price[-1] < real_stock_price).any():
            print("Sell")
        else:
            print("Hold")
        # Visualising the results
        plt.plot(real_stock_price, color = 'red', label = 'Real Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Price')
        plt.title('BTC Price Prediction')
        plt.ylim(0,40000)
        plt.xlabel('Time')
        plt.ylabel('BTC Price')
        plt.legend()
        plt.savefig('btc_price_prediction.png')
    
    def _create_rnn(self, model_name, features_train, results_train, epochs=100, batch_size=32):
        # Initialising the RNN
        self._regressor = Sequential()
        
        # Adding the first LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = self._layer_size, return_sequences = True, input_shape = (features_train.shape[1], 1)))
        self._regressor.add(Dropout(self._dropout))
        
        # Adding a second LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = self._layer_size, return_sequences = True))
        self._regressor.add(Dropout(self._dropout))
        
        # Adding a third LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = self._layer_size, return_sequences = True))
        self._regressor.add(Dropout(self._dropout))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = self._layer_size))
        self._regressor.add(Dropout(self._dropout))
        
        # Adding the output layer
        self._regressor.add(Dense(units = 1))
        
        # Compiling the RNN
        self._regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Fitting the RNN to the Training set
        self._regressor.fit(features_train, results_train, epochs = epochs, batch_size = batch_size)
        self._regressor.save(model_name)
