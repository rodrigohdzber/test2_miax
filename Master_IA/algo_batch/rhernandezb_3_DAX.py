# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:29:48 2021

@author: Rodrigo
"""
"""
En el siguiente algoritmo, utilizo un sharpe con un cruce de medias.
Realizo la media del sharpe de todos los componentes de hace 1 mes y medio y
me quedo con los 5 mayores, esos 5 los "holdeo" aguanto y les aplico un cruce de medias(50 y 100)
para intentar miximizar el riesgo y maximizar el retorno.
"""
import requests
import json
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from datetime import date
from api_bme import APIBMEHandler
from datetime import datetime

algo_tag = "rhernandezb_algo3"
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

ah_ibex = ApiHandler(market = "DAX")
ticker_master = ah_ibex.get_ticker_master()
df = ah_ibex.get_all_close()
benchmark = ah_ibex.get_close_benchmark()
df = ah_ibex.components(df)

def MME(data,periodo):
    mme_df = data.ewm(alpha = 2/(periodo+1), adjust= False).mean()
    return mme_df

def MACD(data,periodo1,periodo2,periodo3):
    mme_rapida = MME(data,periodo1)
    mme_lenta = MME(data,periodo2)
    signal_rapida = mme_rapida-mme_lenta
    signal_lenta = MME(signal_rapida,periodo3)
    macd_hist = signal_rapida-signal_lenta
    return signal_rapida,signal_lenta,macd_hist


def indice_fuerza(datos,volumen):
    fuerza = datos.diff()*volumen
    mme_fuerza_rapida = MME(fuerza,2)
    mme_fuerza_lenta = MME(fuerza,13)
    return fuerza, mme_fuerza_rapida

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
ultimo_dia_sharpe = sharpe_ibex.iloc[-30:,:].mean()
ordenar = sorted(ultimo_dia_sharpe)
maximos_sharpe = ordenar[-5:]
minimo_maximo = min(maximos_sharpe)
posiciones_ok = np.where(ultimo_dia_sharpe >= minimo_maximo)
p = np.array(posiciones_ok)
p = p[0,:]
activos_seleccionados = pd.DataFrame(df.iloc[:,p])

holdeo = ["EONGn.DE","LINI.DE", "MRCG.DE", "MUVGn.DE", "RWEG.DE"]

df_holdeo =df.loc[:,holdeo]
df_holdeo_sin_na = df_holdeo.dropna()

mme_50 = MME(df_holdeo_sin_na,50)
mme_100 = MME(df_holdeo_sin_na,100)
mme_p = df_holdeo_sin_na.rolling(50).mean()
mme_pr = df_holdeo_sin_na.rolling(100).mean()
mme_ew = df_holdeo_sin_na.ewm(50).mean()
mme_eww = df_holdeo_sin_na.ewm(100).mean()


df_medias = pd.DataFrame(index = df_holdeo_sin_na.index)
df_medias["short_EONGn.DE"] = MME(df_holdeo_sin_na.iloc[:,0],50)
df_medias["long_EONGn.DE"] = MME(df_holdeo_sin_na.iloc[:,0],100)
df_medias["short_LINI.DE"] = MME(df_holdeo_sin_na.iloc[:,1],50)
df_medias["long_LINI.DE"] = MME(df_holdeo_sin_na.iloc[:,1],100)
df_medias["short_MRCG.DE"] = MME(df_holdeo_sin_na.iloc[:,2],50)
df_medias["long_MRCG.DE"] = MME(df_holdeo_sin_na.iloc[:,2],100)
df_medias["short_MUVGn.DE"] = MME(df_holdeo_sin_na.iloc[:,3],50)
df_medias["long_MUVGn.DE"] = MME(df_holdeo_sin_na.iloc[:,3],100)
df_medias["short_RWEG.DE"] = MME(df_holdeo_sin_na.iloc[:,4],50)
df_medias["long_RWEG.DE"] =MME(df_holdeo_sin_na.iloc[:,4],100)

df_medias["signals_EONGn.DE"] = np.where(df_medias["short_EONGn.DE"] > df_medias["long_EONGn.DE"], 1, 0)
df_medias["signals_LINI.DE"] = np.where(df_medias["short_LINI.DE"] > df_medias["long_LINI.DE"], 1, 0)
df_medias["signals_MRCG.DE"] = np.where(df_medias["short_MRCG.DE"] > df_medias["long_MRCG.DE"], 1, 0)
df_medias["signals_MUVGn.DE"] = np.where(df_medias["short_MUVGn.DE"] > df_medias["long_MUVGn.DE"], 1, 0)
df_medias["signals_RWEG.DE"] = np.where(df_medias["short_RWEG.DE"] > df_medias["long_RWEG.DE"], 1, 0)

df_medias["position_EONGn.DE"] = df_medias["signals_EONGn.DE"].diff()
df_medias["position_LINI.DE"] = df_medias["signals_LINI.DE"].diff()
df_medias["position_MRCG.DE"] = df_medias["signals_MRCG.DE"].diff()
df_medias["position_MUVGn.DE"] = df_medias["signals_MUVGn.DE"].diff()
df_medias["position_RWEG.DE"] = df_medias["signals_RWEG.DE"].diff()

df_medias["position_EONGn.DE"][df_medias["position_EONGn.DE"] == 0] = None 
df_medias["position_LINI.DE"][df_medias["position_LINI.DE"] == 0] = None 
df_medias["position_MRCG.DE"][df_medias["position_MRCG.DE"] == 0] = None 
df_medias["position_MUVGn.DE"][df_medias["position_MUVGn.DE"] == 0] = None 
df_medias["position_RWEG.DE"][df_medias["position_RWEG.DE"] == 0] = None 

df_medias["position_EONGn.DE"].fillna(method =  "ffill", inplace = True)
df_medias["position_LINI.DE"].fillna(method =  "ffill", inplace = True)
df_medias["position_MRCG.DE"].fillna(method =  "ffill", inplace = True)
df_medias["position_MUVGn.DE"].fillna(method =  "ffill", inplace = True)
df_medias["position_RWEG.DE"].fillna(method =  "ffill", inplace = True)

h = holdeo[0]

"""
Ahora ya lo tengo que mientras estes comprado sea 1
y mientras estas vendido sea -1
entonces, si la ultima fila es 1 se pone el ticker en la lista
si es distinto a uno no se pone en la lista
"""

lista_tickers_con_pesos = []
if df_medias.iloc[-1,15] == 1:
    lista_tickers_con_pesos.append(holdeo[0])
  
if df_medias.iloc[-1,16] == 1:
    lista_tickers_con_pesos.append(holdeo[1])
    
if df_medias.iloc[-1,17] == 1:
    lista_tickers_con_pesos.append(holdeo[2])
    
if df_medias.iloc[-1,18] == 1:
    lista_tickers_con_pesos.append(holdeo[3])

if df_medias.iloc[-1,19] == 1:
    lista_tickers_con_pesos.append(holdeo[4])

else:
    print("ningun activo tiene peso")


def gen_alloc_data(ticker, alloc):
    return {'ticker': ticker,
            'alloc': alloc}


#ahora cuando aparezca un 1 en la ultima fila(que cada día habrá una nueva)
#queremos que en el allocation ese activo aparezca con 1/5
class algo_sharpe_medias:
    
    def __init__(self, market, algo_tag, rebal_period):
        self.market = market
        self.algo_tag = algo_tag
        self.rebal_period = rebal_period
        
        self.ah = APIBME = APIBMEHandler(market = self.market, algo_tag = 'rhernandezb_algo3')
        
    
    def dayly_proc(self):
        #aqui hay que poner el if today == wednesday o algo asi
        data = activos_seleccionados
        data_today = activos_seleccionados.iloc[-1,:]
        #print(data_today.loc[lista_tickers_con_pesos])
        date_ = data_today.name
        print(lista_tickers_con_pesos)
        #da igual hacerlo con date_ que con hoy tendria que cambiar la funcion
        #en vez de post_alocc send alloc
        hoy = date.today().strftime('%Y-%m-%d')
        print(date_)
        print(hoy)
        print(date)
        w = 1/len(lista_tickers_con_pesos)
        print(len(lista_tickers_con_pesos))
        print(w)
        alloc_list = [gen_alloc_data(tck, w) for tck in lista_tickers_con_pesos] 
        #en vez de data_today.index ponemos la lista a la que iremos añadiendo
        #los tickers dependiendo de si hay 1 o -1
        self.ah.post_alloc(hoy,alloc_list)  #hay que poner hoy en todos.

algo_1 = algo_sharpe_medias(market = "DAX", algo_tag = algo_tag, rebal_period = 7) 
z = algo_1.dayly_proc()

file = open("C:/Users/Rodrigo/MASTER_IA/algo_batch/log_ejecuciones.txt", "a")
file.write("algo_3_DAX se ha ejecutado a las " + str(datetime.now()) + "\n")
file.close()

