import requests
import os
import urllib.parse
import csv
import pandas as pd
 
class CryptoCompare:
    def __init__(self, env_name='CRYPTO_COMPARE_API_KEY'):
        self._api_key = os.environ.get(env_name)
        
    def get_daily_history(self, base, quote, limit=2000, last_time=None):
        base_url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        params = {
            'fsym': base,
            'tsym': quote,
            'limit': limit
        }
        if last_time is not None:
            params['toTs'] = last_time
            
        params['api_key'] = self._api_key
            
        url = base_url + "?" + urllib.parse.urlencode(params)
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data['Response'] == 'Success':
                return self._convert_to_dataframe(data)
            else:
                print("Error: ", data['Response'])
                return None
        else:
            print("Error: ", response.status_code)
            return None
        
    def _convert_to_dataframe(self, data):
        ohlvc_list = []
        for data_point in data['Data']['Data']:
            ohlvc_dict = {
                'time': pd.to_datetime(data_point['time'], unit='s'), 
                'open': data_point['open'],
                'high': data_point['high'],
                'low': data_point['low'],
                'close': data_point['close'],
                'volume': data_point['volumeto']
            }
            ohlvc_list.append(ohlvc_dict)
            
        df = pd.DataFrame(ohlvc_list)
        df.set_index('time', inplace=True)
        return df