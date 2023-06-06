import unittest
import sys
sys.path.append('..')
import data_source.crypto_compare as cc
import models.ohlcv as ohlcv

class TestCryptoCompare(unittest.TestCase):
    def test_get_daily_history(self):
        cc_api = cc.CryptoCompare()
        ohlcv_list = cc_api.get_daily_history('BTC', 'USDT', limit=5)
        self.assertNotEqual(ohlcv_list, None)
            
        
if __name__ == '__main__':
    unittest.main()