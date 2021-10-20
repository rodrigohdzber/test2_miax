# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:19:27 2021

@author: Rodrigo
"""
"""
En este algoritmo utilizo el ratio de sharpe para crear un EW.
El algoritmo hace la media del ratio de sharpe de las ultimas dos semanas de todos los componentes
del índice y se queda con los 5 que más ratio tengan. Asigna un peso igual a los 5 y rebalancea cada
dos semanas haciendo este proceso anterior.
"""
import pandas as pd
import requests, json
import datetime
import os
from datetime import datetime
from datetime import date

class ApiHandler:
    def __init__(self, market):  #aqui tengo que poner también algo_tag y abajo self.algo_tag...
        self.market = market
        self.competi = "mia_7"
        self.user_key = 'AIzaSyAN7SczbqGIBnnYeW5rQ0Op-TnpoRXUwHw'
        self.url_base = 'https://miax-gateway-jog4ew3z3q-ew.a.run.app'
    
    def get_ticker_master(self): #obtenemos los tickermaster
        url = f"{self.url_base}/data/ticker_master"
        self.competi
        headers = {'Content-Type': 'application/json'}
        params = {'competi': self.competi,
          'market': self.market,
          'key': self.user_key}
        response = requests.get(url, params)
        tk_master = response.json()
        maestro_df = pd.DataFrame(tk_master["master"])
        return maestro_df
    
    def get_close_data(self, ticker):    #para sacar el benchmark poner como ticker benchmark
        url2 = f'{self.url_base}/data/time_series'
        params = {'market': self.market,
          'key': self.user_key,
          'ticker': ticker}
        response = requests.get(url2, params)
        tk_data = response.json()
        series_data = pd.read_json(tk_data, typ = "series")
        return series_data
    
    def get_close_benchmark(self):
        url2 = f'{self.url_base}/data/time_series'
        params = {'market': self.market,
          'key': self.user_key,
          'ticker': 'benchmark',}
        response = requests.get(url2, params)
        tk_data = response.json()
        series_data = pd.read_json(tk_data, typ = "series")
        return series_data
    
    def get_all_close(self):
        df = pd.DataFrame({
            tck : ah_ibex.get_close_data(tck)
            for tck in ticker_master.loc[:,"ticker"].to_list()
        })
        return df
    
    def update_all_close(self):
        self.df_all_close = self.get_all_close()
        
    def components(self,close_data):    #puedo complementar las otras funciones para que salga todo
        df_x = close_data.iloc[-1,:].dropna()
        last = df_x.index.to_list()
        df = close_data.loc[:,last]
        return df
    
    def send_alloc(self, iday, algo_tag, allocation):
        url = f'{self.url_base}/participants/allocation'
        url_auth = f'{url}?key={self.user_key}'

        str_date = iday.strftime('%Y-%m-%d')
        params = {
            'competi': self.competi,
            'algo_tag': algo_tag,
            'market': self.market,
            'date': str_date,
            'allocation': allocation
        }
        #print(json.dumps(params))
        response = requests.post(url_auth, data=json.dumps(params))
        print (response.json())
    
    def post_alloc(self, str_date, algo_tag, alloc_list):
        
        url = f'{self.url_base}/participants/allocation'
        
        url_auth = f'{url}?key={self.user_key}'
        
        data = {
            
            'competi': self.competi,
            
            'algo_tag': algo_tag,
            
            'market': self.market,
            
            'date': str_date,
            
            'allocation': alloc_list,
            
        }
        
        response = requests.post(url_auth, data=json.dumps(data))
        
        print(response.json())
        
    def run_backtest(self):
        url = f'{self.url_base}/participants/exec_algo'
        url_auth = f'{url}?key={self.user_key}'
        params = {
            'competi': self.competi,
            'algo_tag': algo_tag,  #self.algo_tag
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
    
    def components(self,close_data):    #puedo complementar las otras funciones para que salga todo
        df_x = close_data.iloc[-1,:].dropna()
        last = df_x.index.to_list()
        df = close_data.loc[:,last]
        return df
    
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

ah_ibex = ApiHandler(market = "EUROSTOXX")
ticker_master = ah_ibex.get_ticker_master()
df = ah_ibex.get_all_close()
benchmark = ah_ibex.get_close_benchmark()
df = ah_ibex.components(df)

import numpy as np

def ratio_sharpe(df,benchmark2,ventana):
    retornos = np.log(df).diff()
    rent_benchmark = np.log(benchmark2).diff()
        
    var_benchmark = rent_benchmark.rolling(ventana).var()
        
    volatilidad  = pd.DataFrame(index = df.index)
    
    for i in retornos:
        volatilidad[i] = retornos[i].rolling(ventana).std()
        
    sharpe_activos = retornos / volatilidad
        
    return sharpe_activos

sharpe_ibex = ratio_sharpe(df,benchmark,30)
ultimo_dia_sharpe = sharpe_ibex.iloc[-10:,:].mean()
ordenar = sorted(ultimo_dia_sharpe)
maximos_sharpe = ordenar[-5:]
minimo_maximo = min(maximos_sharpe)
posiciones_ok = np.where(ultimo_dia_sharpe >= minimo_maximo)
p = np.array(posiciones_ok)
p = p[0,:]
activos_seleccionados = pd.DataFrame(df.iloc[:,p])

def gen_alloc_data(ticker, alloc):
    return {'ticker': ticker,
            'alloc': alloc}

algo_tag = "rhernandezb_algo2"

fechas_reb = ["2021-09-01","2021-09-15","2021-09-29","2021-10-13",
               "2021-10-27","2021-11-10","2021-11-24","2021-12-08","2021-12-22",
               "2021-01-12", "2021-01-26", "2021-02-09", "2021-02-23", "2021-03-09"]

class algo_sharpe:
    
    def __init__(self, market, algo_tag, rebal_period):
        self.market = market
        self.algo_tag = algo_tag
        self.rebal_period = rebal_period
        
        self.ah = ah_ibex = ApiHandler(market = self.market)
        
    
    def dayly_proc(self):
        #aqui hay que poner el if today == wednesday o algo asi
        hoy = date.today().strftime('%Y-%m-%d')
        for fecha in fechas_reb:
            print(fecha)
            if fecha == hoy:
                print("hoy se ejecuta")
                data = activos_seleccionados
                data_today = activos_seleccionados.iloc[-1,:]
                print(data_today)
                date_ = data_today.name
                #da igual hacerlo con date_ que con hoy tendria que cambiar la funcion
                #en vez de post_alocc send alloc
                hoy = date.today().strftime('%Y-%m-%d')
                print(date_)
                print(hoy)
                print(date)
                w = 1/data_today.shape[0]
                print(data_today.shape[0])
                print(w)
                alloc_list = [gen_alloc_data(tck, w) for tck in data_today.index]
                self.ah.post_alloc(hoy,algo_tag,alloc_list)  #hay que poner hoy en todos.
            
            else:
                print("hoy no toca")

algo_1 = algo_sharpe(market = "EUROSTOXX", algo_tag = algo_tag, rebal_period = 7) 
z = algo_1.dayly_proc()

file = open("C:/Users/Rodrigo/MASTER_IA/algo_batch/log_ejecuciones.txt", "a")
file.write("algo_2_EUROSTOXX se ha ejecutado a las " + str(datetime.now()) + "\n")
file.close()


