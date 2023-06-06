import argparse
from predictors.btc_ltsm import BtcLtsm

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='BTC Price Prediction')
    parser.add_argument('--update', action='store_true', help='Update the dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()

    btc_ltsm = BtcLtsm()
    if args.update:
        btc_ltsm.update_dataset()
    if args.train:
        btc_ltsm.train()
    if args.test:
        btc_ltsm.load()
        btc_ltsm.test_model()