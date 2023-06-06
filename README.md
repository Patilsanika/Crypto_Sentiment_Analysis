
# Crypto Currency Sentiment Analysis Using Text Abstraction

Our project aims to provide suggestions to the users whether they should buy, sell or hold their share
of cryptocurrencies. To do this we are using sentiment analysis with the help of text abstraction. This project uses a Long Short-Term Memory (LSTM) neural network to predict the price of Bitcoin in US dollars. The LSTM model is implemented using the Keras library in Python. we aim to provide a better solution that
will be applicable for multiple cryptocurrencies with the best possible accuracy.




## Tech Stack

Full project is coded in python language.

Python libraries used are matplotlib, pandas, numpy, sklearn, keras

Nltk library used for sentiment analysis of extracted text.



## APIs Used

Cryptocompare api to fetch data about cryptocurrency prices.

Newsapi client for python to get news data about crypto currency

## Usage

To use the project, you will need to install the necessary dependencies using pip. You can do this by running the following command:

pip install -r requirements.txt

Once the dependencies are installed, you can use the project to predict Bitcoin prices by running the deep_crypto.py script with the appropriate command-line arguments. 

The available arguments are:

--update: Update the dataset with the latest Bitcoin price data from the CryptoCompare API.

--train: Train the LSTM model using the updated data set.

--test: Test the LSTM model on the test data set and visualize the results in btc_price_prediction.png.

For example, to update the dataset, train and test the LSTM model, you can run the following command:
python deep_crypto.py --update --train --test
