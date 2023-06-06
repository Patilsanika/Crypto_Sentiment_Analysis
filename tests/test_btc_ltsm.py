import unittest
import os
import sys
sys.path.append('..')
import predictors.btc_ltsm

class TestBtcLtsn(unittest.TestCase):
    def test_update_dataset(self):
        btc_ltsm = predictors.btc_ltsm.BtcLtsm(limit=50)
        btc_ltsm.update_dataset()
        self.assertTrue(os.path.exists('btc_price_train.csv'))
        self.assertTrue(os.path.exists('btc_price_test.csv'))
            
        
if __name__ == '__main__':
    unittest.main()