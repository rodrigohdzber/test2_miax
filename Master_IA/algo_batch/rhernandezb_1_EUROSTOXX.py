# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:27:49 2021

@author: Rodrigo
"""

"""
En este algoritmo utilizare el modelo de markowitz para minimizar el riesgo y maximizar el retorno.
Utilizo distintos modelos como el mean absolute desviation y el modelo CVar(conditional value at risk)
maximizando el portfolio con el mínimo riesgo posible.
"""

import requests
import json
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from datetime import date
import plotly.graph_objs as go
import plotly.offline as py
import cvxpy as cv
from datetime import datetime

from api_bme import APIBMEHandler

APIBME = APIBMEHandler('EUROSTOXX', 'rhernandezb_algo1')
maestro_df = APIBME.get_ticker_master()
data_close, data_high, data_low, data_open, data_vol = APIBME.get_data()
benchmark = APIBME.get_close_data_ticker('benchmark')
data_close = APIBME.components(data_close)

data_high = APIBME.components(data_high)
data_low = APIBME.components(data_low)
data_open = APIBME.components(data_open)
data_vol = APIBME.components(data_vol)

returns=data_close.pct_change().dropna()
#almacenaremos los nombres de los activos para el grafico final comparando los pesos de todos los modelos
tickers=returns.columns.values 

py.init_notebook_mode() 

def grafico_frontera(frontera, titulo, riesgo, sharpe=None):
    riesgo='Riesgo - ' + str(riesgo)
    trace1 = go.Scatter(
        x=frontera['port_risk'],
        y=frontera['port_ret'],
        name='Frontera',
        mode='markers',
        marker=dict(
            #size = 16,
            color = frontera['port_sharpe'].values, 
            colorscale = 'viridis',
            #reversescale=True,
            showscale=True
        )
    )
    
    if sharpe is None:
        data = [trace1]
    else:
        trace2 = go.Scatter(
            x=sharpe[1],
            y=sharpe[0],
            name='Sharpe Ratio',
            mode='markers',
            marker=dict(
                size = 16,
                color = 'RGB(0,192,0)'
            )
        )
        data = [trace1, trace2]

    layout= go.Layout(
        title= titulo,
        autosize=False,
        width=800,
        height=500,        
        xaxis= dict(
            title= riesgo,
            exponentformat='none',
            tickformat="0.4f"        
        ),yaxis=dict(
            title= 'Retorno',
            exponentformat='none',
            tickformat="0.4f"
        ),
        showlegend= False
    )

    fig=go.Figure(data=data, layout=layout)
    py.iplot(fig, show_link=False)

import matplotlib.pyplot as plt

def grafico_frontera_2(frontera, titulo, riesgo, sharpe=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(frontera.port_risk,frontera.port_ret,c=frontera.port_sharpe,cmap='viridis')
    plt.title(str(titulo))
    plt.xlabel('Riesgo - '+ str(riesgo))
    plt.ylabel('Returns')
    plt.colorbar()    
    if sharpe != None:
        plt.scatter(sharpe[1],sharpe[0],color='g',s=200)  
        
#Calculo los pesos para un solo portfolio
#creando los inputs del modelo de media varianza
mu = np.array(np.mean(returns.values,axis=0), ndmin=2)
sigma = np.cov(returns.values.T)
w = cv.Variable(mu.shape[1])
g1 = cv.Parameter(nonneg=True)
g1.value = 1
ret = mu @ w
risk = cv.quad_form(w, sigma)

#definiendo la funcion objetivo
prob = cv.Problem(cv.Maximize(ret - g1*risk),
                   [cv.sum(w) == 1,
                    w >= 0.01])#esta restriccion para que cada activo tenga como minimo 1%
#resolviendo el problema
prob.solve()
weights = np.array(w.value, ndmin=2)

#almacenando los datos del portafolio
port_ret = (mu @ weights.T).item()
port_risk = np.sqrt(weights @ sigma @ weights.T).item()
Portafolio = pd.DataFrame(data=weights,columns=tickers)

#Automatizo la funcion para los distintos niveles de lambda
#Defino la funcion de frontera effciente media varianza
def frontera_MVO(returns):
    mu = np.array(np.mean(returns,axis=0), ndmin=2)
    sigma = np.cov(returns.T)
    
    w = cv.Variable((mu.shape[1],1))
    g = cv.Parameter(nonneg=True)

    ret = mu @ w
    risk = cv.quad_form(w,sigma)

    port_ret=[]
    port_risk=[]
    port_sharpe=[]
    portafolio={'port_ret':port_ret,
                'port_risk':port_risk,
                'port_sharpe':port_sharpe}

    gs = np.arange(0,10,0.1)
    for i in gs:
        g.value=i
        prob = cv.Problem(cv.Maximize(ret-g*risk),
                           [cv.sum(w) == 1,
                            w >=0.01])
        prob.solve()
        weights=np.array(w.value, ndmin=2)
        try:
            port_ret.append((mu @ weights).item() * 252)
            port_risk.append(np.sqrt(weights.T @ sigma @ weights * 252).item())
            port_sharpe.append((mu @ weights).item()/np.sqrt(weights.T @ sigma @ weights).item())
        except:
            continue
    return pd.DataFrame(portafolio)

frontera_1=frontera_MVO(returns.values)
grafico_frontera(frontera_1, 'Optimización con Markowitz', 'Desviación Estándar')
grafico_frontera_2(frontera_1, 'Optimización con Markowitz', 'Desviación Estándar')

#Maximizo Sharpe y grafico
# Modelo Sharpe Ratio Media Varianza

#defino los inputs
mu=np.array(np.mean(returns,axis=0), ndmin=2)
sigma=np.cov(returns.T)
w = cv.Variable(mu.shape[1])
k = cv.Variable(1)
rf=cv.Parameter(nonneg=True)
rf.value=0
u=np.ones((1,mu.shape[1]))*rf

#defino el problema, funcion objetivo y reestricciones
prob = cv.Problem(cv.Minimize(cv.quad_form(w,sigma)),
               [(mu-u) @ w == 1,
               w >= 0,
               k >= 0,
               w >= 0.01*k, #para que el peso minimo sea 0.01%
               cv.sum(w) == k])
#se resuelve el problema
prob.solve(solver=cv.ECOS)
w_MV = np.array(w.value/k.value, ndmin=2)

#almacenando los datos
SR_MV = []
SR_MV.append([(mu @ w_MV.T * 252).item()])
SR_MV.append([np.sqrt(w_MV @ sigma @ w_MV.T * 252).item()])

#graficp la frontera con el ratio de sharpe
grafico_frontera(frontera_1, 'Optimización con Markowitz', 'Desviación Estándar', SR_MV)
grafico_frontera_2(frontera_1, 'Optimización con Markowitz', 'Desviación Estándar', SR_MV)

#En esta parte utilizo el modelo MAD (mean absolute desviation) de  Konno y Yamazaki para maximizar el retorno del portfolio y minimizar el riesgo
#defino la funcipn de frontera eficiente media MAD

def MAD(returns):
    return np.mean(np.absolute(returns - np.mean(returns,axis=0)), axis=0).item()

def frontera_MAD(returns):
    w = cv.Variable(returns.shape[1])
    Y = cv.Variable(returns.shape[0])
    mu = np.array(np.mean(returns,axis=0), ndmin=2)
    u = np.ones((len(returns), 1)) * mu
    T = cv.Parameter(nonneg=True)
    T.value = len(returns)
    a = returns - u
    risk = cv.sum(Y)/T
    
    port_ret=[]
    port_risk=[]
    port_sharpe=[]
    portafolio={'port_ret':port_ret,
                'port_risk':port_risk,
                'port_sharpe':port_sharpe}
    mus=np.arange(0, np.max(mu), np.max(mu)/100)
    for i in mus:
        prob = cv.Problem(cv.Minimize(risk),
                            [mu @ w>=i,
                             a @ w >= -Y,
                             a @ w <= Y,
                             Y >= 0,
                             cv.sum(w) == 1,
                             w >= 0.01])
        prob.solve(solver=cv.ECOS)
        weights = np.array(w.value, ndmin=2)
        try:
            port_ret.append((mu @ weights.T).item()*252)
            port_risk.append(MAD(returns @ weights.T)*252)
            port_sharpe.append((mu @ weights.T).item()/MAD(returns @ weights.T))
        except:
            continue
    return pd.DataFrame(portafolio)

#Calculo la frontera eficiente MAD y grafico
frontera_2=frontera_MAD(returns.values)
grafico_frontera(frontera_2, 'Optimización con MAD', 'MAD')
grafico_frontera_2(frontera_2, 'Optimización con MAD', 'MAD')

#Hago el Ratio Retorno Riesgo(media MAD) que trata de maximizar el rendimiento por unidad de riesgo
#Modelo Sharpe Ratio MAD

#definiendo los inputs
mu = np.array(np.mean(returns,axis=0), ndmin=2)
w = cv.Variable(returns.shape[1])
Y = cv.Variable(returns.shape[0])
k = cv.Variable(1)
rf = cv.Parameter(nonneg=True)
rf.value = 0
T = cv.Parameter(nonneg=True)
T.value = returns.shape[0]
risk = sum(Y)/T
u = np.ones((len(returns.values),1))*mu
a = returns.values - u

#defino el problema, funcion objetivo y reestricciones
prob = cv.Problem(cv.Minimize(risk),
                            [a @ w >= -Y,
                             a @ w <= Y,
                             Y >= 0,
                             cv.sum(w) == k,
                             mu @ w - rf * k == 1,
                             w >= 0.01*k, #restriccion adicional para pesos mayores a 1%
                             w >= 0])

#resolvo el problema
prob.solve(solver=cv.ECOS)
w_MAD = np.array(w.value/k.value, ndmin=2)

#almaceno los datos
SR_MAD = []
SR_MAD.append([(mu @ w_MAD.T * 252).item()])
SR_MAD.append([ MAD(returns.values @ w_MAD.T) * 252 ])

#grafico la frontera con el ratio de sharpe
grafico_frontera(frontera_2, 'Optimización con MAD', 'MAD', SR_MAD)
grafico_frontera_2(frontera_2, 'Optimización con MAD', 'MAD', SR_MAD)

#Modelo media CVaR, trato de maximizar el retorno minimizando el riesgo
#Defino la funcion de frontera eficente media CVaR Historico

def VaR_Hist(returns, alpha):
    sorted_returns = np.sort(returns, axis=0)
    index = int((1-alpha) * len(sorted_returns))
    return np.abs(sorted_returns[index]).item()

def CVaR_Hist(returns, alpha):
    sorted_returns = np.sort(returns, axis=0)
    index = int((1-alpha) * len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    return np.abs(sum_var / index).item()

def frontera_CVaR_Hist(returns,alpha1):
    mu = np.array(np.mean(returns,axis=0), ndmin=2)
    w = cv.Variable(returns.shape[1])
    n = cv.Parameter(nonneg=True)
    n.value=returns.shape[0]
    VaR=cv.Variable(1)
    alpha=cv.Parameter(nonneg=True)
    alpha.value=alpha1
    X=returns @ w
    Z = cv.Variable(returns.shape[0])
    risk=VaR+1/((1-alpha)*n)*cv.sum(Z)
    port_ret=[]
    port_risk=[]
    port_sharpe=[]
    portafolio={'port_ret':port_ret,
                'port_risk':port_risk,
                'port_sharpe':port_sharpe}
    mus=np.arange(0,np.max(mu),np.max(mu)/100)

    for i in mus:
        prob = cv.Problem(cv.Minimize(risk),
                            [mu @ w>=i,
                             Z >= 0,
                             Z >= -X-VaR,
                             cv.sum(w) == 1,
                             w >= 0.01])
        prob.solve(solver=cv.ECOS)
        weights = np.array(w.value, ndmin=2)
        try:
            port_ret.append((mu @ weights.T).item() *252)
            port_risk.append(CVaR_Hist(returns @ weights.T,alpha1)*np.sqrt(252))
            port_sharpe.append((mu @ weights.T).item()/CVaR_Hist(returns @ weights.T,alpha1))
        except:
            continue
    return pd.DataFrame(portafolio)

#Calculo la frontera eficiente MAD y grafico
frontera_3=frontera_CVaR_Hist(returns.values,0.99)
grafico_frontera(frontera_3, 'Optimización con CVaR Histórico', 'CVaR Histórico')
grafico_frontera_2(frontera_3, 'Optimización con CVaR Histórico', 'CVaR Histórico')

#Aqui trato de maximizar el rendimiento por unidad de riesgo
#Modelo Sharpe Ratio CVaR Histórico

#definiendo los inputs
mu = np.array(np.mean(returns,axis=0), ndmin=2)
w = cv.Variable(returns.shape[1])
n = cv.Parameter(nonneg=True)
n.value = returns.shape[0]
VaR = cv.Variable(1)
alpha = cv.Parameter(nonneg=True)
alpha.value=0.99
Z = cv.Variable(returns.shape[0])
risk = VaR+1/((1-alpha)*n)*cv.sum(Z)
k = cv.Variable(1)
rf = cv.Parameter(nonneg=True)
rf.value=0
X = returns.values @ w

#defino el problema, funcion objetivo y reestricciones
prob = cv.Problem(cv.Minimize(risk),
                    [Z >= 0,
                     Z >= -X - VaR,
                     mu @ w - rf * k == 1,
                     cv.sum(w) == k,
                     w >= 0.01*k, #pesos minimos de 1%
                     w >= 0,
                     k>=0])

#resolv el problema
prob.solve(solver=cv.ECOS)
w_CVaR_Hist=np.array(w.value/k.value, ndmin=2)

#almaceno los datos
SR_CVaR_Hist=[]
SR_CVaR_Hist.append([(mu @ w_CVaR_Hist.T*252).item()])
SR_CVaR_Hist.append([(CVaR_Hist(returns.values @ w_CVaR_Hist.T,0.99)*np.sqrt(252))])

#grafico la frontera con el ratio de sharpe
grafico_frontera(frontera_3, 'Optimización con CVaR Histórico', 'CVaR Histórico',SR_CVaR_Hist)
grafico_frontera_2(frontera_3, 'Optimización con CVaR Histórico', 'CVaR Histórico',SR_CVaR_Hist)

#El modelo de CVaR anterior fue basado en el CVaR Histórico (datos reales históricos), sin embargo tambien se podria basar en datos simulados mediante una simulacion de montecarlo, para ello vamos a definir primero unas funciones que nos generen datos correlacionados (en este caso asumiremos que tienen distribucion normal) que tengan la media y varianza de los datos historicos

def MC_Corr_Sample(returns, N):
    mu=np.array(np.mean(returns,axis=0), ndmin=2)
    cols = returns.shape[1]               
    np.random.seed(0)
    observations = np.random.normal(0, 1, (cols, N)) 
    cov_matrix = np.cov(returns.T)   

    Chol = np.linalg.cholesky(cov_matrix) # Descomposicion de Cholesky 
    
    sam_eq_mean = Chol.dot(observations)             
    samples = sam_eq_mean.transpose() + mu 
    return samples

def MC_Sample(returns, N):
    mu=np.array(np.mean(returns), ndmin=2)
    sd=np.array(np.std(returns), ndmin=2)
    np.random.seed(0)
    observations = np.random.randn(N, 1)
    sam_eq_mean = observations*sd            
    samples = sam_eq_mean + mu      
    return samples

#Con estas funciones, defino unas funciones similares a las definidas para el CVaR historico para calcular la frontera eficiente y el ratio de riesgo retorno del CVaR Montecarlo
# Defino la funcion de frontera effciente media CVaR Montecarlo
def VaR_MC(returns, alpha, N):
    returns=MC_Sample(returns, N)
    sorted_returns = np.sort(returns, axis=0)
    index = int((1-alpha) * len(sorted_returns))
    return np.abs(sorted_returns[index]).item()

def CVaR_MC(returns, alpha, N):
    returns=MC_Sample(returns, N)
    sorted_returns = np.sort(returns, axis=0)
    index = int((1-alpha) * len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    return np.abs(sum_var / index).item()

def frontera_CVaR_MC(returns,alpha1, N):
    mu=np.array(np.mean(returns,axis=0), ndmin=2)
    returns_MC=MC_Corr_Sample(returns, N)
    w = cv.Variable(returns_MC.shape[1])
    n = cv.Parameter(nonneg=True)
    n.value=returns_MC.shape[0]
    VaR=cv.Variable(1)
    alpha=cv.Parameter(nonneg=True)
    alpha.value=alpha1
    X=returns_MC @ w
    Z = cv.Variable(returns_MC.shape[0])
    ret=mu @ w
    risk=VaR+1/((1-alpha)*n)*cv.sum(Z)

    port_ret=[]
    port_risk=[]
    port_sharpe=[]
    portafolio={'port_ret':port_ret,
                'port_risk':port_risk,
                'port_sharpe':port_sharpe}
    mus=np.arange(0,np.max(mu),np.max(mu)/100)
    for i in mus:
        prob = cv.Problem(cv.Minimize(risk),
                            [ret>=i,
                             Z >= 0,
                             Z >= -X-VaR,
                             cv.sum(w) == 1,
                             w >= 0.01])
        prob.solve(solver=cv.ECOS)
        weights=np.array(w.value, ndmin=2)
        try:
            port_ret.append((mu @ weights.T).item()*252)
            port_risk.append(CVaR_MC(returns @ weights.T,alpha1,N)*np.sqrt(252))
            port_sharpe.append((mu @ weights.T).item()/CVaR_MC(returns @ weights.T,alpha1,N))
        except:
            continue
    return pd.DataFrame(portafolio)

#Grafico comparativo de los 3 modelos y grafico

trace1 = go.Bar(
    x=tickers,
    y=w_MV.tolist()[0],
    name='Variance',
    )
trace2 = go.Bar(
    x=tickers,
    y=w_MAD.tolist()[0],
    name='MAD',
    )
trace3 = go.Bar(
    x=tickers,
    y=w_CVaR_Hist.tolist()[0],
    name='CVaR Hist',
    )

data = [trace1, trace2, trace3]
layout = go.Layout(
    autosize=False,
    width=900,
    height=500,
    title= 'Pesos por Modelo',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, show_link=False)

###################################################
#Grafico


n_activos=returns.shape[1]
plt.figure(figsize=(12, 6))
ind = np.arange(n_activos)
width = 0.2
plt.bar(ind + -1*width, w_MV.tolist()[0], width, color='b', bottom=0)
plt.bar(ind + 0*width, w_MAD.tolist()[0], width, color='orange', bottom=0)
plt.bar(ind + 1*width, w_CVaR_Hist.tolist()[0], width, color='green', bottom=0)

plt.title('Pesos por Modelo')
plt.xticks((ind + width / 3),tickers)
plt.legend(('Variance', 'MAD', 'CVaR Hist', 'CVaR MC'))
#plt.show()

def gen_alloc_data(ticker, alloc):
    return {'ticker': ticker,
            'alloc': alloc}
tickers = data_close.columns
tickers = tickers.to_series()
alloc = Portafolio.iloc[-1,:].values

hoy = date.today().strftime('%Y-%m-%d')

allocation = [gen_alloc_data(tickers[i], alloc[i]) for i in np.arange(0,data_close.shape[1])]
APIBME.post_alloc(hoy,allocation)

file = open("C:/Users/Rodrigo/MASTER_IA/algo_batch/log_ejecuciones.txt", "a")
file.write("algo_1_EUROSTOXX se ha ejecutado a las " + str(datetime.now()) + "\n")
file.close()
