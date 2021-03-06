                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                RO DEL PLOT PARA IR VIENDO TODAS LAS ACCIONES.

grafico = function(ticker){
  df_cada_accion  <- ibex_df[ibex_df$name == ticker,]  
  
  df_cada_accion$Beneficios_Acumulados  <-cumsum(df_cada_accion$beneficio) #sumamos por acumulacion
  plot(df_cada_accion$Beneficios_Acumulados,type = "l", xlab = "date", ylab = "Beneficios_Acumulados",main = ticker)
}

for (x in 1:length(vector1)) {
  x = vector1[x]
  grafico(x)

}
  
#Ejercicio 7
#Partiendo del algoritmo de mechas anterior, añade el parámetro price_departure.
#Para cada activo, cada día, si el price_departure es >= 0.75, compra a precio de apertura y vende cuando ocurra el primer de los siguientes eventos: 
#El activo sube 3 céntimos (stop profit)
#El activo cae 10 céntimos (stop loss)
#Si no ocurre ninguno de los anteriores, vende a precio de cierre

#Ojo, habrá días positivos y negativos a la vez, en estos casos, supón que toca primero el stop loss

#El capital que invertimos en cada activo, cada día, debe ser 30.000 €
#La comisión de cada compra y venta será de 0.0003 * capital
#Homogeneiza los datos Ibex_data y price_departure (utiliza solo las fechas que existan en ambos DF)
#Comprueba que tienes al menos 30 datos para hacer los cálculos, antes de aplicar el filtro del price_departure (si no es así descarta el activo)


price_departure = read.csv("../data/price_departures.csv", sep = ",")
library(reshape2)
price_departure <- melt(price_departure,id.vars =names(price_departure)[1],measure.vars = names(price_departure)[2:length(price_departure)])
price_departure <- price_departure[!is.na(price_departure$value),]
names(price_departure) = c("date", "name", "price")

ibex_df_new = inner_join(ibex_df,price_departure)

ibex_tickers_n = ibex_df_new %>%              
  select(name) %>%
  group_by(name) %>%
  summarise(n_operaciones = n())

cond_n = (ibex_tickers_n$n_operaciones < num_min_days)         
acciones_no_validas <-ibex_tickers_n[cond_n,"name"]   
print(acciones_no_validas)                        

acciones_malas = ibex_df_new$name %in% acciones_no_validas  
ibex_df_new = ibex_df_new[!acciones_malas,]                

ibex_df_2 = ibex_df_new

ibex_df_new$loss = ibex_df_new$open - ibex_df_new$low       #orquilla inferior
ibex_df_new$profit = ibex_df_new$high - ibex_df_new$open    #orquilla superior
ibex_df_new$cond_loss = ifelse(ibex_df_new$loss >= stop_loss,TRUE,FALSE)       #si las perdidas son mayores que el stop true si no false
ibex_df_new$cond_profit = ifelse(ibex_df_new$profit >= stop_profit,TRUE,FALSE)
#Cacular cuantas acciones se pueden comprar cada dia con 30000 $
ibex_df_new$Num_Acc = (capital_invertido*(1-ccoste)) %/% ibex_df_new$open     #1-ccoste es lo mismo que capital*ccoste
ibex_df_new$Num_Acc2 = (capital_invertido*(1-ccoste)) %/% ibex_df_new$open
#capital_invertido - (capital_invertido*ccoste)
#necesito calcular cuanto dinero se queda sin invertir
ibex_df_new$rest = (capital_invertido - (ibex_df_new$open*ibex_df_new$Num_Acc)) - capital_invertido*ccoste   #calculamos el resto que le quedaria #no se si hay que restas la comision #escuchar audio a bermejo
ibex_df_new$rest2 = (capital_invertido - (ibex_df_new$open*ibex_df_new$Num_Acc))
#ahora habra crear las condiciones que pueden ocurrir

cond1_n = !(ibex_df_new$cond_loss) & !(ibex_df_new$cond_profit) #no toca ambos limites
cond2_n = (ibex_df_new$cond_loss) &(!(ibex_df_new$cond_profit)) #toca el limite de perdidas
cond3_n = (!ibex_df_new$cond_loss) &(ibex_df_new$cond_profit)   #toca el limite de ganancias
cond4_n = (ibex_df_new$cond_loss & ibex_df_new$cond_profit)     #toca ambos limites suponemos que toca primero el stop




ibex_df_new$precio_de_venta = 0

ibex_df_new[cond1_n,"precio_de_venta"] = ibex_df_new[cond1_n,"close"]    #
ibex_df_new[cond2_n, "precio_de_venta"] = ibex_df_new[cond2_n,"open"] - stop_loss     
ibex_df_new[cond3_n, "precio_de_venta"] = ibex_df_new[cond3_n,"open"] + stop_profit
ibex_df_new[cond4_n, "precio_de_venta"] = ibex_df_new[cond4_n,"open"] - stop_loss

ibex_df_new$cierre <- ibex_df_new$Num_Acc * ibex_df_new$precio_de_venta*(1-ccoste)
ibex_df_new$beneficio = ibex_df_new$rest + ibex_df_new$cierre - capital_invertido
ibex_df_new$diapositivo = ibex_df_new$beneficio > 0

cond_soltar = ibex_df_new$price >= 0.75
ibex_df_new  <- ibex_df_new[cond_soltar,]

beneficio_acumulado_new = ibex_df_new %>%
  group_by(name) %>%
  summarise(sum(beneficio),mean(beneficio),n(),mean(loss),mean(profit),100*sum(diapositivo)/n(),100*(1-sum(diapositivo)/n()))

names(beneficio_acumulado_new) <- c('Name','Beneficio Total','Beneficio Promedio','Numero de Operaciones','Promedio Horquilla Inferior','Promedio Horquilla Superior',
                                'Porcentaje Dias Postivos','Porcentaje Dias Negativos')


df_entregable_2 = beneficio_acumulado_new

vector_n = ibex_df_new %>% select(name) %>% group_by(name) %>% summarise()
vector1_n = vector_n[,1]
vector1_n = sapply(vector1_n, as.character)




grafico_n = function(ticker){
  df_cada_accion_n  <- ibex_df_new[ibex_df_new$name == ticker,]  
  
  df_cada_accion_n$Beneficios_Acumulados  <-cumsum(df_cada_accion_n$beneficio) #sumamos por acumulacion
  plot(df_cada_accion_n$Beneficios_Acumulados,type = "l", xlab = "date", ylab = "Beneficios_Acumulados",main = ticker)
}

for (x in 1:length(vector1_n)) {
  x = vector1_n[x]
  grafico_n(x)
  
}

#Ejercicio 8 
#Optimización de la mecha y el capital por activo
#Utilizar un stop profit, o un stop loss estático, no parece lo más eficiente. 
#Tampoco lo parece el utilizar el mismo capital para todos los activos.
#Objetivo: Partiendo del algoritmo de mechas anterior
#Modifica el capital asignado a cada activo: usa la media de datos de cierre y el 0,5% del volumen.
#Modifica el stop profit de cada activo: utiliza el cuantil 30 de la mecha superior (max – open)
#Modifica el stop loss de cada activo: utiliza el cuantil 80 de la mecha inferior (open – low)

ibex_df_3 = ibex_df_2 %>%
  group_by(name) %>%
  summarise(quantile(ibex_df_2$profit,0.3),quantile(ibex_df_2$loss,0.8),mean(close))

names(ibex_df_3) = c("name", "stop_prof_n", "stop_loss_n", "mean_close")   #como tienen columnas del mismo nombre se puede juntar.
ibex_final = merge(ibex_df_3,ibex_df_2)  

ibex_final$capital = 0.005 * ibex_final$vol * ibex_final$mean_close

ibex_final$cond_loss_new = ifelse(ibex_final$loss >= ibex_final$stop_loss_n,TRUE,FALSE)
ibex_final$cond_profit_new = ifelse(ibex_final$profit >= ibex_final$stop_prof_n,TRUE,FALSE)

ibex_final$num_acc_n = (ibex_final$capital * (1 - ccoste)) %/% ibex_final$open
ibex_final$rest_n = (ibex_final$capital - (ibex_final$open*ibex_final$num_acc_n)*(1+ccoste)) 

cond1_n_n = !(ibex_final$cond_loss_new) & !(ibex_final$cond_profit_new) #no toca ambos limites
cond2_n_n = (ibex_final$cond_loss_new) &(!(ibex_final$cond_profit_new)) #toca el limite de perdidas
cond3_n_n = (!ibex_final$cond_loss_new) &(ibex_final$cond_profit_new)   #toca el limite de ganancias
cond4_n_n = (ibex_final$cond_loss_new & ibex_final$cond_profit_new)     #toca ambos limites suponemos que toca primero el stop


ibex_final$precio_de_venta_n = 0

ibex_final[cond1_n_n,"precio_de_venta_n"] = ibex_final[cond1_n_n,"close"]
ibex_final[cond2_n_n,"precio_de_venta_n"] = ibex_final[cond2_n_n,"open"] - ibex_final[cond2_n_n, "stop_loss_n"]  #resta el stop loss nuevo de las filas que cumplan la condicion 2 a las que cumplan la condicion 2 de open
ibex_final[cond3_n_n,"precio_de_venta_n"] = ibex_final[cond3_n_n,"open"] + ibex_final[cond3_n_n, "stop_prof_n"]
ibex_final[cond4_n_n,"precio_de_venta_n"] = ibex_final[cond4_n_n,"open"] - ibex_final[cond4_n_n, "stop_loss_n"]



ibex_final$cierre = (ibex_final$num_acc_n * ibex_final$precio_de_venta_n) * (1-ccoste)
ibex_final$beneficio = ibex_final$cierre + ibex_final$rest_n - ibex_final$capital
ibex_final$diapositivo = ibex_final$beneficio > 0



cond_soltar_n = ibex_final$price >= 0.75
ibex_final = ibex_final[cond_soltar_n,]


beneficio_acumulado_final = ibex_final %>%
  group_by(name) %>%
  summarise(sum(beneficio),mean(beneficio),round(mean(beneficio/capital),5),n(),mean(loss),mean(profit),100*sum(diapositivo)/n(),100*(1-sum(diapositivo)/n()))

names(beneficio_acumulado_final) <- c('Name','Beneficio Total',"Beneficio promedio","b.euro",'Numero de Operaciones','Promedio Horquilla Inferior','Promedio Horquilla Superior',
                                      'Porcentaje Dias Postivos','Porcentaje Dias Negativos')


df_entregable_final = beneficio_acumulado_final

grafico_n_n = function(ticker){
  df_cada_accion_n_n  <- ibex_final[ibex_final$name == ticker,]  
  
  df_cada_accion_n_n$Beneficios_Acumulados  <-cumsum(df_cada_accion_n_n$beneficio) #sumamos por acumulacion
  plot(df_cada_accion_n_n$Beneficios_Acumulados,type = "l", xlab = "date", ylab = "Beneficios_Acumulados",main = ticker)
}

for (x in 1:length(vector1_n)) {
  x = vector1_n[x]
  grafico_n_n(x)
  
}

