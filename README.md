
# Crypto Currency Sentiment Analysis Using Text Abstraction

Our project aims to provide suggestions to the users whether they should buy, sell or hold their share
of cryptocurrencies. To do this we are using sentiment analysis with the help of text abstraction. This project uses a Long Short-Term Memory (LSTM) neural network to predict the price of Bitcoin in US dollars. The LSTM model is implemented using the Keras library in Python. we aim to provide a better solution that
will be applicable for multiple cryptocurrencies with the best possible accuracy.




## Tech Stack

Full project is coded in python language.

Python libraries used are matplotlib, pandas, numpy, sklearn, keras

Nltk library used for sentiment analysis of extracted text.

## FlowGraph 
![Screenshot 2023-06-06 162838](https://github.com/Patilsanika/Crypto_Sentiment_Analysis/assets/86789929/37d73b25-9ec2-4e94-9a8c-dccfdcade3db)


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

## Results

Result 1
![ss1](https://github.com/Patilsanika/Crypto_Sentiment_Analysis/assets/86789929/2aacf999-5c77-4c68-9a7f-d52da5c89a71)

![graph1](https://github.com/Patilsanika/Crypto_Sentiment_Analysis/assets/86789929/89b8035c-ec26-44c3-a498-ad2f8e829065)
