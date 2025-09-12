'''
Code to run the population model presented in REF.
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sys
import os
import openpyxl
import time
from datetime import datetime
import shutil
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image


'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------'''
""".   ## Functions for parameter reading...      """
'''----------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''

def read_excel_save_dictionary(excel_file):
    dictionary = {}

    try:
        workbook = openpyxl.load_workbook(excel_file)
        sheet = workbook.active
    except Exception as e:
        print(f"Error opening the Excel file: {e}")
        return None

    for row in sheet.iter_rows(min_row=2, values_only=True):
        if len(row) >= 3 and row[1] is not None and row[2] is not None:
            dictionary[row[1]] = row[2]

    workbook.close()
    return dictionary
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------'''
""".   ## Functions for the population dynamic.      """
'''----------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------------------'''
def kappaA(mA,FmaxA):
  return mA+FmaxA

def BA(X,PjA):
  Bj = X[:,1]
  return PjA*Bj

def kappaJ(mj,Fmaxj,PjA):
  return Fmaxj+PjA +mj


def rhoJ(X,rj,KA):
  BA = X[:,0]
  Bj = X[:,1]
  return rj*Psi(1-(Bj+BA)/KA)

def Psi(x):
  return 0*(x<0) + 1*(x>1) + x*(x>0)*(x<1)

def rJ(E,P,scenario,type_of_compliance,eta,Pbase,Sub,params):
  # params = (rJ_NonHarvesting,rJ_Compliance,rJ_NonCompliance,exponent)
  #params_rj = (2.58, 2.58, 0, 1)
  if(scenario=='non-harvesting'):
    return params[0] *np.ones_like(E)
  else:
    if(type_of_compliance=='full-compliance'):
      return params[0]*(1-responseToPrice(eta,Pbase,P+Sub)) +params[1]*responseToPrice(eta,Pbase,P+Sub)
    elif(type_of_compliance=='full-non-compliance'):
      return params[0]*(1-responseToPrice(eta,Pbase,P)) +params[2]*responseToPrice(eta,Pbase,P)
    elif(type_of_compliance=='power-law'):
      return params[0]*(1-responseToPrice(eta,Pbase,P+Sub))+ responseToPrice(eta,Pbase,P+Sub)*params[1]+ responseToPrice(eta,Pbase,P)*((params[2]-params[1])*(1-E)**params[3])# params[3] = 1 = funcion lineal; <1 no lineal

def FmaxA(E,P,scenario,type_of_compliance,eta,Pbase,Sub,params):
  # params = (FmaxACompliance,FmaxNonCompliance,exponent)
  #params_FmaxA = (0.25,1,1)
  if(scenario=='non-harvesting'):
    return np.zeros_like(E)
  else:
    if(type_of_compliance=='full-compliance'):
      return params[0]*responseToPrice(eta,Pbase,P+Sub)
    elif(type_of_compliance=='full-non-compliance'):
      return params[1]*responseToPrice(eta,Pbase,P)
    elif(type_of_compliance=='power-law'):
      return responseToPrice(eta,Pbase,P+Sub)*params[0]*E + responseToPrice(eta,Pbase,P)*((params[1]-params[0])*(1-E)**params[2])  # params[2] = 1 = funcion lineal; <1 no lineal

def Fmaxj(E,P,scenario,type_of_compliance,eta,Pbase,Sub,params):
  # params = (FmaxACompliance,FmaxNonCompliance,exponent)
  #params_FmaxJ = (0,0.25,1)
  if(scenario=='non-harvesting'):
    return np.zeros_like(E)
  else:
    if(type_of_compliance=='full-compliance'):
      return params[0]*responseToPrice(eta,Pbase,P+Sub)
    elif(type_of_compliance=='full-non-compliance'):
      return params[1]*responseToPrice(eta,Pbase,P)
    elif(type_of_compliance=='power-law'):
      return responseToPrice(eta,Pbase,P)*(params[0] + (params[1]-params[0])*(1-E)**params[2])


def simula_precio(precio_actual,mu,sigma,dt,Ntray):
  dB = np.random.normal(0,np.sqrt(dt),size=[Ntray])
  return precio_actual*np.exp((mu-0.5*sigma**2)*dt + sigma*dB)


def responseToPrice(eta,P0,P):
  # se anula bajo P0 aproximadamente
  return 1 / (1+np.exp(-eta*(P-P0)))

"""## Funciones para la dinámica de $E$"""

def betas(X,P,E,KA,Pmin,Pmax,TauU,Sub,eta,bar_b1,bar_b2):
  beta_11=(X[:,0]+X[:,1])/KA
  beta_12= 1- responseToPrice(eta,Pmin,P) + responseToPrice(eta,Pmax,P)
  beta_13= (1-E)/TauU
  beta_21 = 1- beta_11
  beta_22 = responseToPrice(eta,Pmin,P+Sub)
  beta_23 = E/TauU
  out = np.zeros_like(X)
  out[:,0] = (beta_11 + beta_12+1)*beta_13+bar_b1
  out[:,1] =(beta_21 + beta_22+1)*beta_23+bar_b2
  return out


def muE(E,b1,b2,dt):
  return E*np.exp(-(b1+b2)*dt) + b2*(1-np.exp(-(b1+b2)*dt))/(b1+b2)

def sigmaE(sigma2,b1,b2,dt):
  return np.sqrt(sigma2*(1-np.exp(-2*(b1+b2)*dt))/(2*(b1+b2)))

def proyectar01(x):
  return -x*(x<0) + (2-x)*(x>1) + x*(x>0)*(x<1)

def proyectarDeltaT(x,dt):
  return dt*(x<dt) + (1-dt)*(x>1-dt) + x*(x>dt)*(x<1-dt)



def Saltos(N,liminf,limsup):
    return np.random.uniform(low=liminf, high=limsup, size=N)

def phi(X,xi,tipo,parametros):
  if(tipo=='potencia'):
    return parametros[0]*(X**parametros[1])*xi
  else:
    if (tipo == 'trozos'): #dos
      return (X<parametros[2])*(parametros[0]*(X**parametros[1])*xi) + (X>parametros[2])*(parametros[3]*(X**parametros[4])*xi)
    else:
      return np.zeros_like(X)

def simBer(p):
  return 1*(np.random.uniform()<=p)


"""# Modelo Kelp"""

# Parametros del Modelo Poblacional
def main():
  
  try:
    parametersFile = sys.argv[1]
    yearsNoHarvesting = int(sys.argv[2])
    yearsHarvesting = int(sys.argv[3])
    numberOfTrajectories = int(sys.argv[4])
    seed = int(sys.argv[5])
    data_to_save = int(sys.argv[6])
  except Exception as err:
    print('Please excecute as: python3 kelp.py parameters_file_xls yearsNoHarvesting yearsHarvesting numberOfTrajectories Random_Seed Data_to_save(0 none / 1 final / 2 complete)')
    print(f"Unexpected {err=}, {type(err)=}")
    return
  
  parameters = read_excel_save_dictionary(parametersFile)

  if parameters:
      print("Parameters loaded:")
      print(parameters)
      print(parameters.keys())
  else:
      print("There was an issue processing the parameters file. Simulation stopped.")
      return

    # Parámetros de Simulación
  if yearsNoHarvesting:
    TNH = yearsNoHarvesting
  else:
    print("You should indicate the number of years without harvesting.")
    return
  
  if yearsHarvesting:
    TH = yearsHarvesting
  else:
    print("You should indicate the number of years without harvesting.")
    return
  
  if numberOfTrajectories:
    Ntray = numberOfTrajectories
  else:
    print("You should indicate the number of trajectories.")
    return
  

  KA = parameters['KA']
  mj = parameters['mj']
  mA = parameters['mA'] 
  PjA = parameters['PjA']  
  


  # params_rj = (rJ_NonHarvesting,rJ_Compliance,rJ_NonCompliance,exponent)
  params_rj = (parameters['rJ_NonHarvesting'] ,parameters['rJ_Compliance'] , parameters['rJ_NonCompliance'] , parameters['rJ_exponent'] )

  # params_FmaxA = (FmaxACompliance,FmaxNonCompliance,exponent)
  params_FmaxA = (parameters['FmaxA_Compliance'] ,parameters['FmaxA_NonCompliance'] , parameters['FmaxA_exponent'])

  # params_FmaxJ = (FmaxACompliance,FmaxNonCompliance,exponent)
  params_FmaxJ = (parameters['FmaxJ_Compliance'] ,parameters['FmaxJ_NonCompliance'] , parameters['FmaxJ_exponent'])

  # Parámetros dinámica de precios
  meanP = parameters['meanP'] 
  desvP = parameters['desvP'] 
  tasa_base = parameters['base_rate']  # al menos la inflación promedio entre 1991 y 2022

  # Umbrales de precios
  Pini = parameters['Pini'] 
  Pbase = parameters['Pbase'] 
  Pmin = parameters['Pmin'] 
  Pmax = parameters['Pmax'] 
  Sub = parameters['Sub'] 

  # Parámetro de Elasticidad de respuesta al precio
  eta= parameters['eta'] 

  # Parámetros dinámica para E.
  TauU = parameters['TauU'] 
  betabar1 = parameters['betabar1'] 
  betabar2 = parameters['betabar2'] 

  # Sigmas (=0, si no hay ruido Browniano)
  sE = parameters['sE'] #desviacion estándar
  sXA =  parameters['sXA'] 
  sXj = parameters['sXj'] 


  TasaEventos =  parameters['event_rate'] 
  jump_liminf =  parameters['jump_liminf'] 
  jump_limsup =  parameters['jump_limsup'] 
 
  tipo_respuesta_a = parameters['response_type_a']
  parametros_respuesta_a = np.array([float(parameters['coef_response_a_1']),float(parameters['power_response_a_1']),float(parameters['phase_response_a']),float(parameters['coef_response_a_2']),float(parameters['power_response_a_2'])])
  tipo_respuesta_j = parameters['response_type_j']
  parametros_respuesta_j = np.array([float(parameters['coef_response_j_1']),float(parameters['power_response_j_1']),float(parameters['phase_response_j']),float(parameters['coef_response_j_2']),float(parameters['power_response_j_2'])])

  # Umbral para la población
  population_threshold = parameters['population_threshold']


  print('Starting simulation...')

  sim_start = time.time()
  np.random.seed(seed)
  X0 = np.random.uniform(0,KA,size=(Ntray,2))
  P0 = Pini*np.ones(Ntray)
  E0 = np.random.uniform(0.05,0.95,size=(Ntray))


  ####
  #### AQUI EMPIEZA LA SIMULACIÓN

  nSteps=365
  N1 = nSteps*TNH
  N2 = nSteps*TH
  dt = 1/nSteps
  T = TNH+TH
  N = N1+N2

  t = np.arange(0,T+dt,dt)
  P = np.zeros((N2+1,Ntray))
  PbaseH = np.zeros((N2+1))
  PminH = np.zeros((N2+1))
  PmaxH = np.zeros((N2+1))
  SubH = np.zeros((N2+1))

  E = np.zeros((N2+1,Ntray))
  E_c = np.zeros((N2+1,Ntray))
  E_n = np.zeros((N2+1,Ntray))


  XNH = np.zeros((N1+1,Ntray,2))
  X = np.zeros((N2+1,Ntray,2))
  X_c = np.zeros((N2+1,Ntray,2))
  X_n = np.zeros((N2+1,Ntray,2))

  TasaExtracion = np.zeros((N2+1,Ntray,2))
  TasaExtracion_c = np.zeros((N2+1,Ntray,2))
  TasaExtracion_n = np.zeros((N2+1,Ntray,2))

  TasaNacH = np.zeros((N2+1,Ntray))
  TasaNacH_c = np.zeros((N2+1,Ntray))
  TasaNacH_n = np.zeros((N2+1,Ntray))


  B = np.zeros((N2+1,Ntray,2))
  B_c = np.zeros((N2+1,Ntray,2))
  B_n = np.zeros((N2+1,Ntray,2))


  XNH[0,:,:]=X0
  rJNH = rJ(E0,P0,'non-harvesting','power-law',eta,Pbase,Sub,params_rj)
  Fmax_Non_Harvesting = np.zeros_like(E0)

  for k in range(1,N1+1):
    expA = np.exp(- 0.5*sXA*sXA*dt + sXA*np.sqrt(dt)*np.random.normal(size=[Ntray]) )
    expJ = np.exp(- 0.5*sXj*sXj*dt + sXj*np.sqrt(dt)*np.random.normal(size=[Ntray]) )
   
    ind_salto = np.random.binomial(1,TasaEventos*dt,size=(Ntray))
    salto = Saltos(Ntray,jump_liminf,jump_limsup)

    XNH[k,:,0] = ( (1-dt*kappaA(mA,Fmax_Non_Harvesting)) * XNH[k-1,:,0] + PjA * dt * XNH[k-1,:,1] )*expA + phi(XNH[k-1,:,0],salto*ind_salto,tipo_respuesta_a,parametros_respuesta_a)
    XNH[k,:,1] = ( (1-dt*kappaJ(mj,Fmax_Non_Harvesting,PjA))*XNH[k-1,:,1] + dt*rhoJ(XNH[k-1,:,:],rJNH,KA)*XNH[k-1,:,0] )*expJ + phi(XNH[k-1,:,1],salto*ind_salto,tipo_respuesta_j,parametros_respuesta_j)

  XnonHarvesting = XNH[-1,:,:]

  E[0,:] = E0
  P[0,:] = P0
  X[0,:,:] = XnonHarvesting
  X_c[0,:,:] = XnonHarvesting
  X_n[0,:,:] = XnonHarvesting


  # FmaxA(E,V,P,scenario,type_of_compliance,eta,Pbase,params):
  # params = (FmaxACompliance,FmaxNonCompliance,exponent)
  # EDIT: eliminamos -Sub en todos Fmax y rJ porque no estamos estudiando la riqueza del pescador, si no sólo su impulso a extraer
  scenario='harvesting'
  TasaExtracion[0,:,0] =  FmaxA(E[0,:],P[0,:],scenario,'power-law',eta,Pbase,Sub,params_FmaxA)
  TasaExtracion[0,:,1] =  Fmaxj(E[0,:],P[0,:],scenario,'power-law',eta,Pbase,Sub,params_FmaxJ)

  TasaExtracion_c[0,:,0] =  FmaxA(E[0,:],P[0,:],scenario,'full-compliance',eta,Pbase,Sub,params_FmaxA)
  TasaExtracion_c[0,:,1] =  Fmaxj(E[0,:],P[0,:],scenario,'full-compliance',eta,Pbase,Sub,params_FmaxJ)

  TasaExtracion_n[0,:,0] =  FmaxA(E[0,:],P[0,:],scenario,'full-non-compliance',eta,Pbase,Sub,params_FmaxA)
  TasaExtracion_n[0,:,1] =  Fmaxj(E[0,:],P[0,:],scenario,'full-non-compliance',eta,Pbase,Sub,params_FmaxJ)

  TasaNacH[0,:] =  rJ(E[0,:],P[0,:],scenario,'power-law',eta,Pbase,Sub,params_rj)
  TasaNacH_c[0,:] =  rJ(E[0,:],P[0,:],scenario,'full-compliance',eta,Pbase,Sub,params_rj)
  TasaNacH_n[0,:] =  rJ(E[0,:],P[0,:],scenario,'full-non-compliance',eta,Pbase,Sub,params_rj)

  PminH[0]=Pmin
  PmaxH[0]=Pmax
  PbaseH[0]=Pbase
  SubH[0]=Sub

  B[0,:,:] = betas(X[0,:,:],P[0,:],E[0,:],KA,Pmin,Pmax,TauU,Sub,eta,betabar1,betabar2)

  ## Recall Definitions
  # def AA(mA,FmaxA):
  # def BA(X,PjA):
  # def AJ(mj,Fmaxj,PjA):
  # def BJ(X,rj,KA):

  for k in range(1,N2+1):
    expA = np.exp( - 0.5*sXA*sXA*dt + sXA*np.sqrt(dt)*np.random.normal(size=[Ntray]) )
    expJ = np.exp( - 0.5*sXj*sXj*dt + sXj*np.sqrt(dt)*np.random.normal(size=[Ntray]) )

   

    ind_salto = np.random.binomial(1,TasaEventos*dt,size=(Ntray))
    salto = Saltos(Ntray,jump_liminf,jump_limsup)

    X[k,:,0] = ( (1-dt*kappaA(mA,TasaExtracion[k-1,:,0]))*X[k-1,:,0] + PjA*dt*X[k-1,:,1] ) * expA + phi(X[k-1,:,0],salto*ind_salto,tipo_respuesta_a,parametros_respuesta_a)
    X[k,:,1] =  ( (1-dt*kappaJ(mj,TasaExtracion[k-1,:,1],PjA))*X[k-1,:,1] + dt*rhoJ(X[k-1,:,:],TasaNacH[k-1,:],KA)*X[k-1,:,0] )*expJ + phi(X[k-1,:,1],salto*ind_salto,tipo_respuesta_j,parametros_respuesta_j)

    X_c[k,:,0] =( (1-dt*kappaA(mA,TasaExtracion_c[k-1,:,0]))*X_c[k-1,:,0] + PjA*dt*X_c[k-1,:,1] ) * expA  + phi(X_c[k-1,:,0],salto*ind_salto,tipo_respuesta_a,parametros_respuesta_a)
    X_c[k,:,1] =( (1-dt*kappaJ(mj,TasaExtracion_c[k-1,:,1],PjA))*X_c[k-1,:,1] + dt*rhoJ(X_c[k-1,:,:],TasaNacH_c[k-1,:],KA)*X_c[k-1,:,0] )*expJ  + phi(X_c[k-1,:,1],salto*ind_salto,tipo_respuesta_j,parametros_respuesta_j)

    X_n[k,:,0] = ( (1-dt*kappaA(mA,TasaExtracion_n[k-1,:,0]))*X_n[k-1,:,0] + PjA*dt*X_n[k-1,:,1] ) * expA  + phi(X_n[k-1,:,0],salto*ind_salto,tipo_respuesta_a,parametros_respuesta_a)
    X_n[k,:,1] =( (1-dt*kappaJ(mj,TasaExtracion_n[k-1,:,1],PjA))*X_n[k-1,:,1] + dt*rhoJ(X_n[k-1,:,:],TasaNacH_n[k-1,:],KA)*X_n[k-1,:,0] )*expJ + phi(X_n[k-1,:,1],salto*ind_salto,tipo_respuesta_j,parametros_respuesta_j)

  # Actualización umbrales (ajuste por la inflación)
    Pmin = Pmin*np.exp(dt*tasa_base)
    Pmax = Pmax*np.exp(dt*tasa_base)
    Pbase = Pbase*np.exp(dt*tasa_base)
    Sub = Sub*np.exp(dt*tasa_base)
    PminH[k]=Pmin
    PmaxH[k]=Pmax
    SubH[k]=Sub
    PbaseH[k]=Pbase
    P[k,:] = simula_precio(P[k-1,:],meanP,desvP,dt,Ntray)


    b=betas(X[k-1,:,:],P[k-1,:],E[k-1,:],KA,Pmin,Pmax,TauU,Sub,eta,betabar1,betabar2)
    b_c=betas(X_c[k-1,:,:],P[k-1,:],E_c[k-1,:],KA,Pmin,Pmax,TauU,Sub,eta,betabar1,betabar2)
    b_n=betas(X_n[k-1,:,:],P[k-1,:],E_n[k-1,:],KA,Pmin,Pmax,TauU,Sub,eta,betabar1,betabar2)

    B[k,:,:]=b
    B_c[k,:,:]=b_c
    B_n[k,:,:]=b_n

    E[k,:] = proyectarDeltaT(E[k-1,:] + (b[:,1]*(1-E[k-1,:])-b[:,0]*E[k-1,:])*dt + sE*np.sqrt(E[k-1,:]*(1-E[k-1,:]))*np.sqrt(dt)*np.random.normal(size=[Ntray]) ,dt)
    E_c[k,:] = proyectarDeltaT(E_c[k-1,:] + (b_c[:,1]*(1-E_c[k-1,:])-b_c[:,0]*E_c[k-1,:])*dt + sE*np.sqrt(E_c[k-1,:]*(1-E_c[k-1,:]))*np.sqrt(dt)*np.random.normal(size=[Ntray]) ,dt)
    E_n[k,:] = proyectarDeltaT(E_n[k-1,:] + (b_n[:,1]*(1-E_n[k-1,:])-b_n[:,0]*E_n[k-1,:])*dt + sE*np.sqrt(E_n[k-1,:]*(1-E_n[k-1,:]))*np.sqrt(dt)*np.random.normal(size=[Ntray]) ,dt)

    TasaExtracion[k,:,0] =   FmaxA(E[k,:],P[k,:],scenario,'power-law',eta,PbaseH[k],SubH[k],params_FmaxA)
    TasaExtracion[k,:,1] =  Fmaxj(E[k,:],P[k,:],scenario,'power-law',eta,PbaseH[k],SubH[k],params_FmaxJ)

    TasaExtracion_c[k,:,0] =   FmaxA(E_c[k,:],P[k,:],scenario,'full-compliance',eta,PbaseH[k],SubH[k],params_FmaxA)
    TasaExtracion_c[k,:,1] =  Fmaxj(E_c[k,:],P[k,:],scenario,'full-compliance',eta,PbaseH[k],SubH[k],params_FmaxJ)

    TasaExtracion_n[k,:,0] =   FmaxA(E_n[k,:],P[k,:],scenario,'full-non-compliance',eta,PbaseH[k],SubH[k],params_FmaxA)
    TasaExtracion_n[k,:,1] =  Fmaxj(E_n[k,:],P[k,:],scenario,'full-non-compliance',eta,PbaseH[k],SubH[k],params_FmaxJ)

    TasaNacH[k,:] =  rJ(E[k,:],P[k,:],scenario,'power-law',eta,PbaseH[k],SubH[k],params_rj)
    TasaNacH_c[k,:] =  rJ(E_c[k,:],P[k,:],scenario,'full-compliance',eta,PbaseH[k],SubH[k],params_rj)
    TasaNacH_n[k,:] =  rJ(E_n[k,:],P[k,:],scenario,'full-non-compliance',eta,PbaseH[k],SubH[k],params_rj)

  sim_end = time.time()
  
  # Elapsed time of the population under the threshold.

  proportionUnderThreshold = np.mean(X[:,:,0]+X[:,:,0] <= population_threshold, axis=1)
  
  simulation_time = sim_end-sim_start
  simulation_hours = int(simulation_time/3600)
  simulation_minutes= int((simulation_time - 3600*simulation_hours)/60)
  simulation_seconds= int(simulation_time - 3600*simulation_hours - 60*simulation_minutes)+1
  simulation_time_msg = 'Simulation completed in ' + str(simulation_hours) + ' hours(s), ' + str(simulation_minutes) +' minute(s) and ' + str(simulation_seconds) +' second(s).'
  print(simulation_time_msg)

  #######################################################################################
  #######################################################################################
  ################################## Plotting and saving
  #######################################################################################
  #######################################################################################
  saving_start = time.time()
  timestamp = datetime.now()
  dirName = parametersFile[0:-5]+'-' + str(timestamp.year)+'-' + str(timestamp.month)+'-' + str(timestamp.day)+'-' + str(timestamp.hour)+'-' + str(timestamp.minute)+'-' + str(timestamp.second)
  os.mkdir(dirName)
  if data_to_save>0:
    if(data_to_save==1):
      np.savez_compressed(dirName+'/data.npz',t=t,tH=t[N1:],X=X[-1,:,:],X_c=X_c[-1,:,:],X_n=X_n[-1,:,:],E=E[-1,:],Nat=TasaNacH[-1,:],Nat_n= TasaNacH_n[-1,:], Nat_c=TasaNacH_c[-1,:],Ext=TasaExtracion[-1,:],Ext_c=TasaExtracion_c[-1,:],Ext_n=TasaExtracion_n[-1,:],P=P,PminH=PminH,PmaxH=PmaxH,SubH=SubH,XNH=XnonHarvesting,put=proportionUnderThreshold,**parameters)
    else:
      np.savez_compressed(dirName+'/data.npz',t=t,tH=t[N1:],X=X,X_c=X_c,X_n=X_n,E=E,Nat=TasaNacH, Nat_n=TasaNacH_n,Nat_c=TasaNacH_c,Ext=TasaExtracion,P=P,PminH=PminH,PmaxH=PmaxH,SubH=SubH,XNH=XNH,put=proportionUnderThreshold,**parameters)
  log_file = dirName + "/simulation_log.txt"
  
  
  filename = os.path.basename(parametersFile)
  shutil.copy(parametersFile,dirName+'/'+filename)
  
    ##############################################################################################################################################
  ################################### Plotting
  ##############################################################################################################################################

  print('Plotting price..')
  tt = t[N1:]
  q = np.arange(0,101,step=10)
  IC_V =  np.percentile(P,q,axis=1)
  fig = plt.figure(figsize=(8,4))
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(tt, IC_V[5,:], label='Median', color='black', alpha=0.9)
  for k in range(5):
    ax.fill_between(tt, IC_V[k,:], IC_V[10-k,:], color='black', alpha=0.1)
  ax.plot(tt,PminH, label='P_min', color='blue', linestyle='--', marker='o' ,markevery=3*365)
  ax.plot(tt,PmaxH,label='P_max', color='red', linestyle='--', marker='x',markevery=3*365)
  ax.set_xlabel('Time [years]')
  ax.set_ylabel('Price')
  ax.legend()
  plt.savefig(dirName+"/price.png", dpi=300)
  plt.close(fig)

  
  ##############################################################################################################################################
  ################################### Plotting Histrograms for population under Non Harvesting
  ##############################################################################################################################################
  print('Plotting Histrograms for population under Non Harvesting...')

  labels = ['XA_NH','XJ_NH']
  tt = t[0:N1+2]
  for label in labels:
    if(label=="XA_NH"):
      Xtotal=XNH[:,:,0]
    if(label=="XJ_NH"):
      Xtotal=XNH[:,:,1]

    K = Xtotal.shape[0]

    minX = np.min(Xtotal)
    maxX = np.max(Xtotal)
    nbins = 200
    dv =  (maxX-minX)/nbins
    b = np.arange(minX,maxX+dv,step =dv)
    h2d = np.zeros((len(b)-1,K))

    for k in range(K):
      hist,bk = np.histogram(Xtotal[k,:],bins=b, density=True )
      h2d[:,k]= hist

    fig, ax = plt.subplots(figsize=(9,4))
    ax.imshow(h2d, extent=[tt[0] , tt[-1],maxX, minX], aspect='auto')
    pos  = ax.invert_yaxis()
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Biomass [g/m2]')
    plt.tight_layout()
    filename = dirName+"/"+label + ".png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)

  tt = t[0:N1+2]
  Xtotal=XNH[:,:,0]+XNH[:,:,1]
  filename=dirName+"/histogramsOverTimeTotalPopulationNonHarvesting.png"
  K = Xtotal.shape[0]
  minX = np.min(Xtotal)
  maxX = np.max(Xtotal)
  nbins = 200
  dv =  (maxX-minX)/nbins
  b = np.arange(minX,maxX+dv,step =dv)
  h2d = np.zeros((len(b)-1,K))

  for k in range(K):
    hist,bk = np.histogram(Xtotal[k,:],bins=b, density=True )
    h2d[:,k]= hist

  fig, ax = plt.subplots(figsize=(9,4))
  ax.imshow(h2d, extent=[tt[0] , tt[-1],maxX, minX], aspect='auto')#, norm=LogNorm(), cmap='plasma')
  pos  = ax.invert_yaxis()
  fig.colorbar(pos, ax=ax)
  ax.set_xlabel('Time [years]')
  ax.set_ylabel('Biomass [g/m2]')
  plt.tight_layout()
  plt.savefig(filename, dpi=300)
  plt.close(fig)

  ##############################################################################################################################################
  ################################### Plotting Histrograms for population under Harvesting
  ##############################################################################################################################################
  print('Plotting Histrograms for population under Harvesting...')

  labels = ['XA','XJ','XA_c','XJ_c','XA_n','XJ_n']
  tt = t[N1:]
  for label in labels:
    if(label=="XA"):
      Xtotal=X[:,:,0]
    if(label=="XA_c"):
      Xtotal=X_c[:,:,0]
    if(label=="XA_n"):
      Xtotal=X_n[:,:,0]
    if(label=="XJ"):
      Xtotal=X[:,:,1]
    if(label=="XJ_c"):
      Xtotal=X_c[:,:,1]
    if(label=="XJ_n"):
      Xtotal=X_n[:,:,1]
    K = Xtotal.shape[0]

    minX = np.min(Xtotal)
    maxX = np.max(Xtotal)
    nbins = 200
    dv =  (maxX-minX)/nbins
    b = np.arange(minX,maxX+dv,step =dv)
    h2d = np.zeros((len(b)-1,K))

    for k in range(K):
      hist,bk = np.histogram(Xtotal[k,:],bins=b, density=True )
      h2d[:,k]= hist

    fig, ax = plt.subplots(figsize=(9,4))
    ax.imshow(h2d, extent=[tt[0] , tt[-1],maxX, minX], aspect='auto')
    pos  = ax.invert_yaxis()
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Biomass [g/m2]')
    plt.tight_layout()
    filename = dirName+"/"+label + ".png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
  


  
  labels = ['X','X_c','X_n']
  tt = t[N1+1:]
  minX = np.min(np.hstack([X[:,:,0]+X[:,:,1],X_c[:,:,0]+X_c[:,:,1],X_n[:,:,0]+X_n[:,:,1]]))
  maxX = np.max(np.hstack([X[:,:,0]+X[:,:,1],X_c[:,:,0]+X_c[:,:,1],X_n[:,:,0]+X_n[:,:,1]]))

  for label in labels:
    if(label=="X"):
      Xtotal=X[:,:,0]+X[:,:,1]
      filename = dirName+"/histogramsOverTimeTotalPopulationDynamicModel.png"
    if(label=="X_c"):
      Xtotal=X_c[:,:,0]+X_c[:,:,1]
      filename = dirName+"/histogramsOverTimeTotalPopulationFullCompliance.png"
    if(label=="X_n"):
      Xtotal=X_n[:,:,0]+X_n[:,:,1]
      filename = dirName+"/histogramsOverTimeTotalPopulationFullNonCompliance.png"
    K = Xtotal.shape[0]

    nbins = 200
    dv =  (maxX-minX)/nbins
    b = np.arange(minX,maxX+dv,step =dv)
    h2d = np.zeros((len(b)-1,K))

    for k in range(K):
      hist,bk = np.histogram(Xtotal[k,:],bins=b, density=True )
      h2d[:,k]= hist

    fig, ax = plt.subplots(figsize=(9,4))
    ax.imshow(h2d, extent=[tt[0] , tt[-1],maxX, minX], aspect='auto')#, norm=LogNorm(), cmap='plasma')
    pos  = ax.invert_yaxis()
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Biomass [g/m2]')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


  
  ##############################################################################################################################################
  ################################### Plotting Rates for population under Harvesting
  ##############################################################################################################################################
  print('Plotting Rates for population under Harvesting...')

  labels = ['E','rJ','extJ','extA']
  tt = t[N1:]
  for label in labels:
    if(label=="E"):
      Xtotal=E
      text_ylabel = 'E'
    if(label=="rJ"):
      Xtotal=TasaNacH
      text_ylabel = 'rJ'
    if(label=="extJ"):
      Xtotal=TasaExtracion[:,:,1]
      text_ylabel = 'Extraction Rate Juveniles'
    if(label=="extA"):
      Xtotal=TasaExtracion[:,:,0]
      text_ylabel = 'Extraction Rate Adults'

    K = Xtotal.shape[0]
    if(label=='rJ'):
      minX = np.min(Xtotal)
      maxX = np.max(Xtotal)
    else:
      minX = 0
      maxX = 1
      
    nbins = 200
    dv =  (maxX-minX)/nbins
    b = np.arange(minX,maxX+dv,step =dv)
    h2d = np.zeros((len(b)-1,K))

    for k in range(K):
      hist,bk = np.histogram(Xtotal[k,:],bins=b, density=True )
      h2d[:,k]= hist

    fig, ax = plt.subplots(figsize=(9,4))
    ax.imshow(h2d, extent=[tt[0] , tt[-1],maxX, minX], aspect='auto')
    pos  = ax.invert_yaxis()
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('Time [years]')
    
    ax.set_ylabel(text_ylabel)
    plt.tight_layout()
    filename = dirName+"/"+label + ".png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)


  ##############################################################################################################################################
  ################################### Plots by year
  ##############################################################################################################################################

  print('Plotting figures by year')
  cutParameter = 1 # 
  alphaParameter = 0.8
  yearsToCompare = 3
  custom_colors = "colorblind"#["blue", "red", "black"]
  sns.color_palette(custom_colors)
  Fill_Parameter = False

  maxAdults= 1.1*np.max(X[-1,:,0])
  maxJuvenile= 1.1*np.max(X[-1,:,1])

  df_nonharvesting =  pd.DataFrame(XNH[0,:,:], columns=['Adult', 'Juvenile'])
  df_nonharvesting['Scenario'] = 'Non harvesting'
  df_nonharvesting['Year'] = 0

  for k in range(1,yearsNoHarvesting+1):
    df_nonharvesting_aux =  pd.DataFrame(XNH[k*nSteps,:,:], columns=['Adult', 'Juvenile'])
    df_nonharvesting_aux['Scenario'] = 'Non harvesting'
    df_nonharvesting_aux['Year'] = k
    df_nonharvesting = pd.concat([df_nonharvesting, df_nonharvesting_aux], ignore_index=True)
    

  df_dyn =  pd.DataFrame(X[0,:,:], columns=['Adult', 'Juvenile'])
  df_dyn['Scenario'] = 'Dynamic'
  df_dyn['Year'] = yearsNoHarvesting

  df_full_c =  pd.DataFrame(X_c[0,:,:], columns=['Adult', 'Juvenile'])
  df_full_c['Scenario'] = 'Full compliance'
  df_full_c['Year'] = yearsNoHarvesting

  df_E = pd.DataFrame(E[0,:], columns=['E'])
  df_E['Year'] = yearsNoHarvesting

  gif_J_vs_A = []
  gif_J_vs_A_fc = []
  gif_E_vs_TP = []
  gif_J_vs_A_scenarios = []

  for k in range(0,yearsHarvesting+1):
    plottingMessage = 'Plotting figures of Year ' +str(yearsNoHarvesting+k)
    print(plottingMessage)
    # We add as reference the final state of the non-harvesting scenario
    df_nonharvesting_aux =  pd.DataFrame(XNH[yearsNoHarvesting*nSteps,:,:], columns=['Adult', 'Juvenile'])
    df_nonharvesting_aux['Scenario'] = 'Non harvesting'
    df_nonharvesting_aux['Year'] = yearsNoHarvesting+ k
    df_nonharvesting = pd.concat([df_nonharvesting, df_nonharvesting_aux], ignore_index=True)

    df_dyn_aux =  pd.DataFrame(X[k*nSteps,:,:], columns=['Adult', 'Juvenile'])
    df_dyn_aux['Scenario'] = 'Dynamic'
    df_dyn_aux['Year'] = yearsNoHarvesting + k
    df_dyn = pd.concat([df_dyn,df_dyn_aux], ignore_index=True)

    df_full_c_aux =  pd.DataFrame(X_c[k*nSteps,:,:], columns=['Adult', 'Juvenile'])
    df_full_c_aux['Scenario'] = 'Full compliance'
    df_full_c_aux['Year'] = yearsNoHarvesting + k
    df_full_c = pd.concat([df_full_c,df_full_c_aux], ignore_index=True)

    df_E_aux = pd.DataFrame(E[k*nSteps,:], columns=['E'])
    df_E_aux['Year'] = yearsNoHarvesting + k
    df_E = pd.concat([df_E,df_E_aux], ignore_index= True)
    df_scenarios = pd.concat([df_dyn_aux, df_full_c_aux,df_nonharvesting_aux], ignore_index=True)

    ###########################################################
    ## Plot for J vs A in dynamic extraction
    plt.figure(figsize=(8, 6))
    ax=sns.kdeplot(data=df_dyn_aux,
                  levels=5,x="Adult", y="Juvenile",cut=cutParameter, 
                  fill=True,  common_norm=False, alpha=alphaParameter )
    plt.grid(True)
    ax.set_xlim(0, maxAdults)
    ax.set_ylim(0, maxJuvenile)
    ax.set_title("Year "+str(yearsNoHarvesting+k))
    fig_title = dirName+'/LevelSetHistogram_Dynamic_year_'+str(yearsNoHarvesting+k)+'.png'
    plt.savefig(fig_title,dpi=400)
    plt.close()
    try:
      img = Image.open(fig_title)
      gif_J_vs_A.append(img.copy())  # Usa .copy() para evitar problemas
    except Exception as e:
      print(f"Error al procesar la imagen {fig_title}: {e}")

###########################################################
    ## Plot for J vs A in full compliance extraction
    plt.figure(figsize=(8, 6))
    ax=sns.kdeplot(data=df_full_c_aux,
                  levels=5,x="Adult", y="Juvenile",cut=cutParameter, 
                  fill=True,  common_norm=False, alpha=alphaParameter )
    plt.grid(True)
    ax.set_xlim(0, maxAdults)
    ax.set_ylim(0, maxJuvenile)
    ax.set_title("Year "+str(yearsNoHarvesting+k))
    fig_title = dirName+'/LevelSetHistogram_full_compliance_year_'+str(yearsNoHarvesting+k)+'.png'
    plt.savefig(fig_title,dpi=400)
    plt.close()
    try:
      img = Image.open(fig_title)
      gif_J_vs_A_fc.append(img.copy())  # Usa .copy() para evitar problemas
    except Exception as e:
      print(f"Error al procesar la imagen {fig_title}: {e}")

    ###########################################################
    ## Plot for J vs A by scenarios
    plt.figure(figsize=(8, 6))
    ax=sns.kdeplot(data=df_scenarios,
                  levels=5,x="Adult", y="Juvenile", hue="Scenario",cut=cutParameter, 
                  fill=Fill_Parameter,  common_norm=False, alpha=alphaParameter )
    sns.move_legend(ax, "lower right")
    plt.grid(True)
    ax.set_xlim(0, maxAdults)
    ax.set_ylim(0, maxJuvenile)
    ax.set_title("Year "+str(yearsNoHarvesting+k))
    fig_title = dirName+'/LevelSetHistogram_Scenarios_year_'+str(yearsNoHarvesting+k)+'.png'
    plt.savefig(fig_title,dpi=400)
    plt.close()
    try:
      img = Image.open(fig_title)
      gif_J_vs_A_scenarios.append(img.copy())  # Usa .copy() para evitar problemas
    except Exception as e:
      print(f"Error al procesar la imagen {fig_title}: {e}")

    ###########################################################
    ## Plot for E vs Total Population
    plt.figure(figsize=(8, 6))
    ax=sns.kdeplot(x=df_dyn_aux['Adult']+df_dyn_aux['Juvenile'], y=df_E_aux['E'],
                  levels=5,cut=cutParameter, 
                  fill=True,  common_norm=False, alpha=alphaParameter )
    plt.grid(True)
    ax.set_xlabel("Total Population")
    ax.set_ylabel("E")
    ax.set_xlim(0, 1.1*KA)
    ax.set_ylim(0, 1)
    ax.set_title("Year "+str(yearsNoHarvesting+k))
    fig_title = dirName+'/LevelSetHistogram_E_vs_TotalPopulation_year_'+str(yearsNoHarvesting+k)+'.png'
    plt.savefig(fig_title,dpi=400)
    plt.close()
    try:
      img = Image.open(fig_title)
      gif_E_vs_TP.append(img.copy())  # Usa .copy() para evitar problemas
    except Exception as e:
      print(f"Error al procesar la imagen {fig_title}: {e}")

  if gif_J_vs_A:
    # Asegúrate de que esta línea esté fuera del bucle, después de que todos los PNGs se hayan generado
    gif_J_vs_A[0].save(
        dirName+'/LevelSetHistogram_J_vs_A.gif',
        save_all=True,
        append_images=gif_J_vs_A[1:],
        duration=500,  # Duración de cada cuadro en milisegundos
        loop=0
    )
    print("¡GIF J vs A creado exitosamente!")
  else:
    print("No se pudieron generar imágenes para el GIF.")
  
  if gif_J_vs_A_fc:
    # Asegúrate de que esta línea esté fuera del bucle, después de que todos los PNGs se hayan generado
    gif_J_vs_A_fc[0].save(
        dirName+'/LevelSetHistogram_J_vs_A_fc.gif',
        save_all=True,
        append_images=gif_J_vs_A_fc[1:],
        duration=500,  # Duración de cada cuadro en milisegundos
        loop=0
    )
    print("¡GIF J vs A full compliance creado exitosamente!")
  else:
    print("No se pudieron generar imágenes para el GIF.")
  
  if gif_E_vs_TP:
    # Asegúrate de que esta línea esté fuera del bucle, después de que todos los PNGs se hayan generado
    gif_E_vs_TP[0].save(
        dirName+'/LevelSetHistogram_E_vs_TP.gif',
        save_all=True,
        append_images=gif_E_vs_TP[1:],
        duration=500,  # Duración de cada cuadro en milisegundos
        loop=0
    )
    print("¡GIF E vs Total Population creado exitosamente!")
  else:
    print("No se pudieron generar imágenes para el GIF.")
  
  if gif_J_vs_A_scenarios:
    # Asegúrate de que esta línea esté fuera del bucle, después de que todos los PNGs se hayan generado
    gif_J_vs_A_scenarios[0].save(
        dirName+'/LevelSetHistogram_J_vs_A_scenarios.gif',
        save_all=True,
        append_images=gif_J_vs_A_scenarios[1:],
        duration=500,  # Duración de cada cuadro en milisegundos
        loop=0
    )
    print("¡GIF J vs A distintos escenarios creado exitosamente!")
  else:
    print("No se pudieron generar imágenes para el GIF.")
  ###############################################################################################################################################
  ################################### Plots to compare between years (to evaluate convergence)
  ###############################################################################################################################################
  plt.figure(figsize=(8, 6))
  ax=sns.kdeplot(data=df_nonharvesting[(df_nonharvesting["Year"]<=yearsNoHarvesting) & (df_nonharvesting["Year"]>yearsNoHarvesting-yearsToCompare)],
                levels=5,x="Adult", y="Juvenile", hue="Year",cut=cutParameter, 
                fill=Fill_Parameter,  common_norm=False, alpha=alphaParameter ,
                palette=custom_colors)
  sns.move_legend(ax, "lower right")
  plt.grid(True)
  fig_title = dirName+'/LevelSetHistogram_non_harvesting_over_time_.png'
  plt.savefig(fig_title,dpi=400)
  plt.close()

  plt.figure(figsize=(8, 6))
  ax=sns.kdeplot(data=df_dyn[df_dyn["Year"]>yearsNoHarvesting+yearsHarvesting-yearsToCompare],
                levels=5,x="Adult", y="Juvenile", hue="Year",cut=cutParameter, 
                fill=Fill_Parameter,  common_norm=False, alpha=alphaParameter,
                palette=custom_colors )
  sns.move_legend(ax, "lower right")
  plt.grid(True)
  fig_title = dirName+'/LevelSetHistogram_Adult_vs_Junenile_Dynamic_over_time.png'
  plt.savefig(fig_title,dpi=400)
  plt.close()

  df_dyn_E = df_dyn
  df_dyn_E["Total Population"] = df_dyn['Adult']+df_dyn['Juvenile']
  df_dyn_E["E"] = df_E["E"]

  plt.figure(figsize=(8, 6))
  ax=sns.kdeplot(data=df_dyn_E[df_dyn_E["Year"]>yearsNoHarvesting+yearsHarvesting-yearsToCompare],
                levels=5,x="Total Population", y="E", hue="Year",cut=cutParameter, 
                fill=Fill_Parameter,  common_norm=False, alpha=alphaParameter ,palette=custom_colors)
  
  sns.move_legend(ax, "lower right")
  plt.grid(True)
  fig_title = dirName+'/LevelSetHistogram_Total_Population_vs_E_Dynamic_over_time.png'
  plt.savefig(fig_title,dpi=400)
  plt.close()
  
  ###############################################################################################################################################
  ################################### Log's writing 
  ###############################################################################################################################################
  

                                    
  saving_end = time.time()
  saving_time = saving_end-saving_start
  saving_hours = int(saving_time/3600)
  saving_minutes= int((saving_time - 3600*saving_hours)/60)
  saving_seconds= int(saving_time - 3600*saving_hours - 60*saving_minutes)+1
  saving_time_msg = 'Data saved in ' + str(saving_hours) + ' hours(s), ' + str(saving_minutes) +' minute(s) and ' + str(saving_seconds) +' second(s).\n'
  
  print(saving_time_msg)



  with open(log_file, "w") as file:
    # Write variables to the file
    file.write(f"Script Version: {sys.argv[0]}\n")
    file.write(f"Parameters File: {parametersFile}\n")
    file.write(f"Years without harvesting: {yearsNoHarvesting}\n")
    file.write(f"Years with harvesting: {yearsHarvesting}\n")
    file.write(f"Number of trajectories: {numberOfTrajectories}\n")
    file.write(f"Seed: {seed}\n")
    if data_to_save>0:
      file.write(f"Data saved:t,tH,X,X_c,X_n,E,Nat,Nat_n, Nat_c,Ext,Ext_c,Ext_n,P,PminH,PmaxH,SubH,XNH,put,params)\n")
    else:
      file.write("No data was saved.\n")
    file.write(simulation_time_msg)
    file.write(saving_time_msg)

if __name__ == "__main__":
  main()
