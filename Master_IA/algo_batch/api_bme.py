# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:16:35 2021

@author: ignac
"""


import requests

import json

import logging

import pandas as pd

logger = logging.getLogger(__name__)

def allocs_to_frame(json_allocations):
    
    alloc_list = []
    
    for json_alloc in json_allocations:
        
        #print(json_alloc)
        
        allocs = pd.DataFrame(json_alloc['allocations'])
        
        allocs.set_index('ticker', inplace=True)
        
        alloc_serie = allocs['alloc']
        
        alloc_serie.name = json_alloc['date'] 
        
        alloc_list.append(alloc_serie)
        
    all_alloc_df = pd.concat(alloc_list, axis=1).T
    
    return all_alloc_df
    
class APIBMEHandler:
        
    def __init__(self, market, algo_tag):
        
        self.url_base = 'https://miax-gateway-jog4ew3z3q-ew.a.run.app'
        
        self.competi = 'mia_7'
        
        self.user_key = 'AIzaSyAN7SczbqGIBnnYeW5rQ0Op-TnpoRXUwHw'
        
        self.market = market
        
        self.algo_tag = algo_tag
         
    def get_ticker_master(self):
        
        url = f'{self.url_base}/data/ticker_master'
        
        params = {'competi': self.competi,
        
                  'market': self.market,
                  
                  'key': self.user_key}
                  
        response = requests.get(url, params)
        
        tk_master = response.json()
        
        maestro_df = pd.DataFrame(tk_master['master'])
        
        return maestro_df
    
    def components(self,close_data):    #puedo complementar las otras funciones para que salga todo
        df_x = close_data.iloc[-1,:].dropna()
        last = df_x.index.to_list()
        df = close_data.loc[:,last]
        return df
        
    def get_close_benchmark(self):
        url2 = f'{self.url_base}/data/time_series'
        params = {'market': self.market,
          'key': self.user_key,
          'ticker': 'benchmark',}
        response = requests.get(url2, params)
        tk_data = response.json()
        series_data = pd.read_json(tk_data, typ = "series")
        return series_data
    
    def get_close_data(self):
        
        maestro_df = self.get_ticker_master()
        
        data_close = {}
        
        for i, data in maestro_df.iterrows():
            
            ticker = data.ticker
            
            logger.info(ticker)
            
            data_close[ticker] = self.get_close_data_ticker(ticker)
            
        data_close = pd.DataFrame(data_close)
        
        return data_close
        
    def get_data(self):
        
        maestro_df = self.get_ticker_master()
        
        data_open = {}
        
        data_high = {}
        
        data_low = {}
        
        data_close = {}
        
        data_vol = {}
        
        for i, data in maestro_df.iterrows():
            
            ticker = data.ticker
            
            logger.info(ticker)
            
            data_aux = self.get_data_ticker(ticker)
            
            data_open[ticker] = data_aux.iloc[:,0]
            
            data_high[ticker] = data_aux.iloc[:,1]
            
            data_low[ticker] = data_aux.iloc[:,2]
            
            data_close[ticker] = data_aux.iloc[:,3]
            
            data_vol[ticker] = data_aux.iloc[:,4]
            


        data_close = pd.DataFrame(data_close)
        
        data_open = pd.DataFrame(data_open)
        
        data_high = pd.DataFrame(data_high)
        
        data_low = pd.DataFrame(data_low)
        
        data_vol = pd.DataFrame(data_vol)
        
        return data_close,data_high,data_low,data_open,data_vol
        
    def get_data_ticker(self, ticker):
        
        url = f'{self.url_base}/data/time_series'
        
        params = {'market': self.market,
        
                  'key': self.user_key,
                  
                  'ticker': ticker,
                  
                  'close': False} 
                  
        response = requests.get(url, params)
        
        tk_data = response.json()
        
        if response.status_code == 200:
            
            df_data = pd.read_json(tk_data, typ='frame')
            
        else: 
            
            print(response.text)
            
        return df_data
        
    def get_close_data_ticker(self, ticker):
        
        url = f'{self.url_base}/data/time_series'
        
        params = {'market': self.market,
        
                  'key': self.user_key,
                  
                  'ticker': ticker} 
                  
        response = requests.get(url, params)
        
        tk_data = response.json()
        
        series_data = pd.read_json(tk_data, typ='series')
        
        return series_data
        
    def get_user_algos(self):
        
        url = f'{self.url_base}/participants/algorithms'
        
        params = {'competi': self.competi,
        
                  'key': self.user_key}
                  
        response = requests.get(url, params)
        
        algos = response.json()
        
        if algos:
            
            algos_df = pd.DataFrame(algos)
            
            print(algos_df.to_string())
            
            return algos_df
            
    def post_alloc(self, str_date, alloc_list):
        
        url = f'{self.url_base}/participants/allocation'
        
        url_auth = f'{url}?key={self.user_key}'
        
        data = {
            
            'competi': self.competi,
            
            'algo_tag': self.algo_tag,
            
            'market': self.market,
            
            'date': str_date,
            
            'allocation': alloc_list,
            
        }
        
        response = requests.post(url_auth, data=json.dumps(data))
        
        print(response.json())
         
    def get_allocs(self):
        
        url = f'{self.url_base}/participants/algo_allocations'
        
        params = {
            
            'key': self.user_key,
            
            'competi': self.competi,
            
            'algo_tag': self.algo_tag,
            
            'market': self.market,
            
        }
        
        response = requests.get(url, params)
        
        print(response.content)
        
        df = allocs_to_frame(response.json())
        
        return df
        
    def run_backtest(self):
        
        url = f'{self.url_base}/participants/exec_algo'
        
        url_auth = f'{url}?key={self.user_key}'
        
        params = {
            
            'competi': self.competi,
            
            'algo_tag': self.algo_tag,
            
            'market': self.market,
            
        }
        
        response = requests.post(url_auth, data=json.dumps(params))
        
        if response.status_code == 200:
            
            exec_data = response.json()
            
            status = exec_data.get('status')
            
            print(status)
            
            res_data = exec_data.get('content')
            
            trades = None
            
            if res_data:
                
                print(pd.Series(res_data['result']))
                
                trades = pd.DataFrame(res_data['trades'])
                
            return res_data
            
        else:
            
            exec_data = dict()
            
            print(response.text)
            
    def delete_allocs(self):
        
        url = f'{self.url_base}/participants/delete_allocations'
        
        url_auth = f'{url}?key={self.user_key}'
        
        params = {
            
            'competi': self.competi,
            
            'algo_tag': self.algo_tag,
            
            'market': self.market,
            
        }
        
        response = requests.post(url_auth, data=json.dumps(params))
        
        print(response)
        

