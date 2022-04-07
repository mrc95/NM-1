#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:57:56 2022

@authors: Marco Lingua, Enrico Bacchetti (Group 5)

"""
#%%

from math import exp, sqrt
import numpy as np
from numpy import log as ln
from numpy import mean
import matplotlib.pyplot as plt
from scipy.stats import norm

T = 5
S0 = 100
N = 60   # frequency in T (months)
r = 0.01
q = 0.05
sigma = 0.15
K = 100
dt=T/N

#%%
"""
TASK 1) - MC plain vanilla call
"""

NSim = 3000 # number of simulations for this task

# void matrix for gbm stock

S = np.zeros(shape=(NSim,N+1))
S[:,0] = S0

# compute stock paths

for i in range(int(N)):
    for j in range(int(NSim)):
        S[j,i+1] = S[j,i]*np.exp( (r-q-sigma**2/2)*dt + sqrt(dt)*sigma*np.random.normal(0,1) )

# vector of payoffs

X = np.zeros(shape=(NSim))

# fill payoff vector

for i in range(int(NSim)):
               X[i] = np.maximum(S[i,-1]-K,0)
               
call1 = mean(X)*np.exp(-r*dt*N) # plain vanilla call simulated price
  
print("Task 1) Point estimate:", call1)
print ("Task 1) MC radius:" , np.std(X*np.exp(-r*dt*N))*1.96/(sqrt(NSim)))
print("Task 1) 95% CI:" , [call1 - np.std(X*np.exp(-r*dt*N))*1.96/(sqrt(NSim)), 
                   call1 + np.std(X*np.exp(-r*dt*N))*1.96/(sqrt(NSim))])
print ("Task 1) MC radius/point estimate:" , np.std(X*np.exp(-r*dt*N))*1.96/(sqrt(NSim))/call1)

#%%
"""
TASK 1) - B&S closed-form solution
"""
d1 = ( ln(S0/K)+(r-q+0.5*sigma**2)*T ) / (sigma*sqrt(T) )
d2 = d1 - sigma * sqrt(T)
N_d1 = norm.cdf(d1)
N_d2 = norm.cdf(d2)

call1_BS = S0*exp(-q*T)*N_d1 - K*exp(-r*T)*N_d2

print("Task 1) B&S closed form:",call1_BS)

#%%
"""
TASK 2) 
"""

gamma = 0.10  # annual probability that holder leaves company

Lambda = -ln(1-gamma)
p_m = 1-exp(-Lambda*1*dt)  # probability that holder leaves with monthly monitoring

call2_BS = call1_BS *((1-p_m)**N) 

print("Task 2) B&S derived closed form:", call2_BS)
        
#%%

"""
TASK 3)
"""

NSim = 2500 # number of simulations

# stores stock stoclastic process

S = np.zeros(shape=(NSim,N+1))
S[:,0] = S0

#  computes stock stoclastic process

for i in range(int(N)):
    for j in range(int(NSim)):
        S[j,i+1] = S[j,i]*np.exp( (r-q-sigma**2/2)*dt + sqrt(dt)*sigma*np.random.normal(0,1) )

# matrix of eso payoff evolution

X_e = np.zeros(shape=(NSim,N)) 

# vector of payoff (both if holder leaves or remains until T=5)
 
X = np.zeros(shape=(NSim))

# at t = 1/12 if holder leaves, cashes in  a payoff >= 0, otherwise the options survives

for j in range(int(NSim)):
    if (np.random.uniform(0,1) < p_m) : # if holder leaves
        X_e[j,0]=np.maximum(S[j,1]-K,0) # cashes in
        X[j] = X_e[j,0]*exp(-r*dt*1) # fill payoff vector with istantaneous discounted payoff
    else:
        X_e[j,0]=S[j,1] # option survives, so the stock prices continues 
  
# from time t=2/12 to end, verify if holder leaves and cashes in or the opposite  
   
for i in range(1,int(N)):
    for j in range(int(NSim)):
        if X_e[j,i-1] == S[j,i] :   # checks if, at  previous time, holder did not leave
            if (np.random.uniform(0,1) < p_m) : # if holder leaves 
                X_e[j,i] = np.maximum(S[j,i+1]-K,0) # cashes in
                X[j] = X_e[j,i]*exp(-r*dt*i) # fill payoff vector with istantaneous discounted payoff
            else:
                X_e[j,i] = S[j,i+1] # option survives, so the stock prices continues 
 
# final period, checks is holder did not leave until then  
   
for j in range(int(NSim)):
     if np.array_equal(X_e[j,-1],S[j,-1]):   
         X[j] = np.maximum(S[j,-1]-K,0)*np.exp(-r*dt*N) #computes standard option payoff
                              
call3 = mean(X) # ESO fair value at t=0

print("Task 3) Point estimate:", call3)
print ("Task 3) MC radius:" , np.std(X*exp(-r*dt*N))*exp(-r*1)*1.96/(sqrt(NSim)))
print("Task 3) 95% CI:" , [ call3 - np.std(X*exp(-r*dt*N))*1.96/(sqrt(NSim)), 
                   call3 + np.std(X*exp(-r*dt*N))*1.96/(sqrt(NSim))])
print ("Task 3) MC radius/point estimate:" , np.std(X*exp(-r*dt*N))*1.96/(sqrt(NSim))/call3) 

#%%
"""
TASK 4)
"""
# this task follows the same logic of the previus, with the only additional condition of good/bad leaver

NSim = 5000

alpha_g = 2 #good leaver multiplier 
alpha_b = 0.7 #bad leaver multiplier 

S = np.zeros(shape=(NSim,N+1))
S[:,0] = S0

for i in range(int(N)):
    for j in range(int(NSim)):
        S[j,i+1] = S[j,i]*np.exp( (r-q-sigma**2/2)*dt + sqrt(dt)*sigma*np.random.normal(0,1) )

# matrix of eso payoff evolution

X_e_gb = np.zeros(shape=(NSim,N))

# vector of payoff (both if holder leaves or remains untile T=5)

X_gb = np.zeros(shape=(NSim))

# at t = 1/12 if holder leaves, cashes in  a payoff >= 0, otherwise the options survives

for j in range(int(NSim)):
    if (np.random.uniform(0,1) < p_m) :
        X_e_gb[j,0]=np.maximum(S[j,1]-K,0) 
        X_gb[j] = X_e_gb[j,0]*exp(-r*dt*1)
    else:
        X_e_gb[j,0]=S[j,1]
        
# from time t=2/12 to end, verify if holder leaves and cashes in or the opposite         
        
for i in range(1,int(N)):
    for j in range(int(NSim)):
        if X_e_gb[j,i-1] == S[j,i] :
            if (np.random.uniform(0,1) < p_m) :
                if (S[j,i+1]>1.5*S0):                 #if good leaver
                   X_e_gb[j,i] = alpha_g*np.maximum(S[j,i+1]-K,0)
                   X_gb[j] = X_e_gb[j,i]*exp(-r*dt*i)
                else:                                 #if bad leaver
                    X_e_gb[j,i] = alpha_b*np.maximum(S[j,i+1]-K,0) 
                    X_gb[j] = X_e_gb[j,i]*exp(-r*dt*i)
            else:
                X_e_gb[j,i] = S[j,i+1]
 
# final period, checks is holder did not leave until then   
               
for j in range(int(NSim)):
     if np.array_equal(X_e_gb[j,-1],S[j,-1]): 
         if (S[j,-1]>1.5*S0):
            X_gb[j] = alpha_g*np.maximum(S[j,-1]-K,0)*np.exp(-r*dt*N)  #if good leaver
         else:
             X_gb[j] = alpha_b*np.maximum(S[j,-1]-K,0)*np.exp(-r*dt*N) #if bad leaver
                               
call4 = mean(X_gb) #ESO fair value at time 0

print("Task 4) Point estimate:", call4)
print ("Task 4) MC radius:" , np.std(X_gb*exp(-r*dt*N))*exp(-r*1)*1.96/(sqrt(NSim)))
print("Task 4) 95% CI:" , [ call4 - np.std(X_gb*exp(-r*dt*N))*1.96/(sqrt(NSim)), 
                   call4 + np.std(X_gb*exp(-r*dt*N))*1.96/(sqrt(NSim))])
print ("Task 4) MC radius/point estimate:" , np.std(X_gb*exp(-r*dt*N))*1.96/(sqrt(NSim))/call4) 

#%%

"""
TASK 5.1)
"""

r_0 = 0.01
r_T = 0.02
T = 5
dt_ = 1/48  # weekly time step
weeks = 240
rho = 0.8
kappa = 0.5
V_0 = 0.0225
theta = 0.04
xi = 0.05
q = 0.05
K = 100

steps = 240    

S_0 = 100

NSim = 2500

# stock, volatility and risk-free initialized structures

S_t = np.zeros((NSim,steps+1))
V_t = np.zeros((NSim,steps+1))
r = np.zeros((1,steps+1))   

r[:,0] = r_0
V_t[:,0] = V_0
S_t[:, 0] = S_0

# creating a bivariate random normal 

means = [0,0]
stdevs = [1, 1]
covs = [[stdevs[0]**2          , stdevs[0]*stdevs[1]*rho], 
            [stdevs[0]*stdevs[1]*rho,           stdevs[1]**2]]

Z = np.random.multivariate_normal(means, covs,    
                 (steps+1, NSim)).T
Z1 = Z[0]
Z2 = Z[1]

# compute rate, volatility and stock processes 

for i in range(1, steps+1):
   
         r[0,i] = r[0,i-1]+(r_T - r_0)/T* dt_
         
         V_t[:,i] = np.maximum(V_t[:,i-1] + 
                kappa * (theta - V_t[:,i-1])* dt_ + 
                xi *  np.sqrt(np.maximum(V_t[:,i-1],0)) * np.sqrt(dt_) * Z2[:,i],0) 
         
         S_t[:,i] = S_t[:,i-1]*np.exp((r[0,i] - q -
                0.5*(V_t[:,i-1]))*dt_ +
                np.sqrt(dt_)*np.sqrt(V_t[:,i-1]) * Z1[:,i])

# computing payoffs and option value

payoff = np.maximum(S_t[:,-1]-K,0)
call_1h = mean(payoff)*exp(-r_T*dt_*steps)  

print("Task 5.1) Point estimate:", call_1h)
print ("Task 5.1) MC radius:" , np.std(payoff*exp(-r_T*dt_*steps))*1.96/(sqrt(NSim)))
print("Task 5.1) 95% CI:" , [call_1h - np.std(payoff*exp(-r_T*dt_*steps))*1.96/(sqrt(NSim)), 
                   call_1h + np.std(payoff*exp(-r_T*dt_*steps))*1.96/(sqrt(NSim))])
print ("Task 5.1)  MC radius/point estimate:" , np.std(payoff*exp(-r_T*dt_*steps))*1.96/(sqrt(NSim))/call_1h)

#%%
# Plots

plt.plot(r[0,:],'b')

plt.ylabel('r(t)')
plt.xlabel('t (weeks)')
plt.title('Interest rate process')

#%%
plt.close()

#%%

for i in range(int(10)):
    plt.plot(S_t[i,:])
    
plt.ylabel('S(t)')
plt.xlabel('t (weeks)')
plt.title('Simulated stock price process')

#%%
plt.close()

#%%
    
for j in range(int(10)):  
    plt.plot(V_t[j,:])

plt.ylabel('V(t)')
plt.xlabel('t (weeks)')
plt.title('Stochastic Volatility process')

#%%
plt.close()

#%%
"""
TASK 5.2)
"""

gamma = 0.10  # annual probability that holder leaves company

Lambda = -np.log(1-gamma)
p_m_h = 1-exp(-Lambda*1*dt_)  # probability that holder leaves with weekly monitoring

call_2h = call_1h*((1-p_m_h)**steps)  # call2 will change every time, since is based on call1, a MC estimate

print("Task 5.2) Point estimate:", call_2h)
print ("Task 5.2) MC radius:" , np.std(payoff*exp(-r_T*dt_*steps))*1.96*((1-p_m_h)**steps) /(sqrt(NSim)))
print("Task 5.1) 95% CI:" , [call_2h - np.std(payoff*exp(-r_T*dt_*steps))*1.96*((1-p_m_h)**steps)/(sqrt(NSim)), 
                   call_2h + np.std(payoff*exp(-r_T*dt_*steps))*1.96*((1-p_m_h)**steps)/(sqrt(NSim))])
print ("Task 5.2)  MC radius/point estimate:" , np.std(payoff*exp(-r_T*dt_*steps)*((1-p_m_h)**steps))*1.96/(sqrt(NSim))/call_1h)

#%%
"""
TASK 5.3)
"""

r_0 = 0.01
r_T = 0.02
T = 5
dt_ = 1/48  # weekly time step
weeks = 240
rho = 0.8
kappa = 0.5
V_0 = 0.0225
theta = 0.04
xi = 0.05
q = 0.05
K = 100

steps = 240    

S_0 = 100

NSim = 2500


gamma = 0.10  # annual probability that holder leaves company

Lambda = -np.log(1-gamma)
p_m_h = 1-exp(-Lambda*1*dt_) 


S_t = np.zeros((NSim,steps+1))
V_t = np.zeros((NSim,steps+1))
r = np.zeros((1,steps+1))   

r[:,0] = r_0
V_t[:,0] = V_0
S_t[:, 0] = S_0

means = [0,0]
stdevs = [1, 1]
covs = [[stdevs[0]**2          , stdevs[0]*stdevs[1]*rho], 
            [stdevs[0]*stdevs[1]*rho,           stdevs[1]**2]]

Z = np.random.multivariate_normal(means, covs,    
                 (steps+1, NSim)).T             
Z1 = Z[0]
Z2 = Z[1]

# compute rate, volatility and stock processes 

for i in range(1, steps+1):
   
         r[0,i] = r[0,i-1]+(r_T - r_0)/T* dt_
         
         V_t[:,i] = np.maximum(V_t[:,i-1] + 
                kappa * (theta - V_t[:,i-1])* dt_ + 
                xi *  np.sqrt(np.maximum(V_t[:,i-1],0)) * np.sqrt(dt_) * Z2[:,i],0) 
         
         S_t[:,i] = S_t[:,i-1]*np.exp((r[0,i] - q -
                0.5*(V_t[:,i-1]))*dt_ +
                np.sqrt(dt_)*np.sqrt(V_t[:,i-1]) * Z1[:,i])
         
                 
X_e_h = np.zeros(shape=(NSim,steps)) #option value matrix

#option final payoff vector, store discounted values if holder leaves before T and at maturity 

X_h = np.zeros(shape=(NSim)) 

# check for the first time step if holder leaves or not

for j in range(int(NSim)):
    if (np.random.uniform(0,1) < p_m_h) :
        X_e_h[j,0]=max(S_t[j,1]-K,0)
        X_h[j] = X_e_h[j,0]*np.exp(-r[0,1]*dt_*1)
    else:
        X_e_h[j,0]=S_t[j,1]
    
 # check for all the other time steps if holder leaves or not
       
for i in range(1,int(steps)):
    for j in range(int(NSim)):
        if X_e_h[j,i-1] == S_t[j,i] :   
            if (np.random.uniform(0,1) < p_m_h) :
                X_e_h[j,i] = max(S_t[j,i+1]-K,0)
                X_h[j] = X_e_h[j,i]*exp(-r[0,i]*dt_*i)
            else:
                X_e_h[j,i] = S_t[j,i+1]

# computes final payoffs and discounts
                
for j in range(int(NSim)):
     if np.array_equal(X_e_h[j,-1],S_t[j,-1]):
         X_h[j] = max(S_t[j,-1]-K,0)*exp(-r[0,-1]*dt_*steps)
         
for i in range(int(20)):
    plt.plot(X_e_h[i,:])   

plt.ylabel('ESO(t)')
plt.xlabel('t (weeks)')
plt.savefig('eso 5.3.png',dpi=300)
                    
call_3h = mean(X_h)

print("Task 5.3) Point estimate:", call_3h)
print ("Task 5.3) MC radius:" , np.std(X_h*exp(-r_T*dt_*steps) )*1.96/(sqrt(NSim)))
print("Task 5.3) 95% CI:" , [call_3h - np.std(X_h*exp(-r_T*dt_*steps) )*1.96/(sqrt(NSim)), 
                   call_3h + np.std(X_h*exp(-r_T*dt_*steps) )*1.96/(sqrt(NSim))])
print ("Task 5.3) MC radius/point estimate:" , np.std(X_h*exp(-r_T*dt_*steps) )*1.96/(sqrt(NSim))/call_3h)
    
#%%

""""
TASK 6) 
"""
r = 0.01

# calibration parameters

u = exp(sigma*sqrt(dt))
d = 1/u
p = (np.exp((r-q)*dt)-d)/(u-d)

S_bin = np.zeros([N+1, N+1])

# stock binomial tree 
   
for i in range(N+1):
    for j in range(i+1):
        S_bin[j,i] = S0*(d**j)*(u**(i-j))
            
call_6 = np.zeros([N+1, N+1])
call_6[:,N] = np.maximum(S_bin[:,N]-K,0)   # payoffs at maturity
        
for i in np.arange(N-1, -1, -1):
    for j in np.arange(0, i+1):
        call_6[j,i] = np.maximum(np.maximum(S_bin[j,i]-K,0),np.exp(-r*dt)*(p*call_6[j,i+1]+(1-p)*call_6[j+1,i+1]))

print("Task 6) American call option:", call_6[0,0])    
 
#%%

"""
TASK 7)
"""

NSim = 150

dt = 1/12
months_1y = 12

# simulates GBM stock paths along the first year 

S_t_1y = np.zeros((NSim,months_1y+1))
S_t_1y[:,0] = S0

for i in range(int(months_1y)):
    for j in range(int(NSim)):
        S_t_1y[j,i+1] = S_t_1y[j,i]*np.exp( (r-q-sigma**2/2)*dt + sqrt(dt)*sigma*np.random.normal(0,1) )
  
# populates each stock binomial trees  
       
S_bin_m = np.zeros([NSim, N-months_1y+1, N-months_1y+1])
  
for k in range(int(NSim)):
    S_bin_m[k,0,0] = S_t_1y[k,-1]
  
for k in range(int(NSim)):    
    for i in range(1,N-months_1y+1):
        for j in range(i+1):
            S_bin_m[k,j,i] = S_bin_m[k,0,0]*(d**j)*(u**(i-j))

# computes each opttion with their own strike

call_7_m = np.zeros([NSim, N-months_1y+1, N-months_1y+1])

for k in range(int(NSim)):  
    call_7_m[k,:,-1] = np.maximum(S_bin_m[k,:,-1]-min(S_t_1y[k,:]),0)

for k in range(int(NSim)):
    for i in np.arange(N-months_1y-1, -1, -1):
        for j in np.arange(0, i+1):
            call_7_m[k,j,i] = np.maximum(np.maximum(S_bin_m[k,j,i]-min(S_t_1y[k,:]),0),
                                         exp(-r*dt)*(p*call_7_m[k,j,i+1]+(1-p)*call_7_m[k,j+1,i+1]))
# eso fair price at time 0
            
call7 = mean(call_7_m[:,0,0])*exp(-r*dt*months_1y)    

print("Task 7) Point estimate:", call7)
print ("Task 7) MC radius:" , np.std(call_7_m[:,0,0])*exp(-r*1)*1.96/(sqrt(NSim)))
print("Task 7) 95% CI:" , [ call7 - np.std(call_7_m[:,0,0])*exp(-r*1)*1.96/(sqrt(NSim)), 
                   call7 + np.std(call_7_m[:,0,0])*exp(-r*1)*1.96/(sqrt(NSim))])
print ("Task 7) MC radius/point estimate:" , np.std(call_7_m[:,0,0])*exp(-r*1)*1.96/(sqrt(NSim))/call7) 

#%%

"""
TASK 8)
"""
NSim = 90

months_1y = 12
dt_ = 1/48

# generates stock price paths

S_t_1y = np.zeros((NSim,months_1y+1))
S_t_1y[:,0] = S0

for i in range(int(months_1y)):
    for j in range(int(NSim)):
        S_t_1y[j,i+1] = S_t_1y[j,i]*np.exp( (r-q-sigma**2/2)*dt 
                                           + sqrt(dt)*sigma*np.random.normal(0,1) )

S_bin_m8 = np.zeros([NSim, N+1, N+1]) #3D srtucture of binomial stock trees

n_cols = 2

enter_prices = np.zeros((NSim, n_cols)) #vectors of maximum prices and relative time index (t_MAX)

# look for maximum values and their time 

for k in range(int(NSim)):
    enter_prices[k,0] = np.amax(S_t_1y[k,:])
    enter_prices[k,1] = np.argmax(S_t_1y[k,:])

# set starting value of each binomial tree equal to the maximum of the simulated path

for k in range(int(NSim)):
    S_bin_m8[k,0,0] = enter_prices[k,0]

# computes stock binomial process

for k in range(int(NSim)):    
    for i in range(1,int(N-enter_prices[k,1]+1)):
        for j in range(i+1):
            S_bin_m8[k,j,i] = S_bin_m8[k,0,0]*(d**j)*(u**(i-j))  

strikes = np.zeros((NSim))  #vectors of strike prices and relative time index
idx_strikes = 0 

# looks for maximum from time zero to time_Max

for k in range(int(NSim)):
    idx_strikes = enter_prices[k,1]
    strikes[k] = np.amin(S_t_1y[k, 0:int(idx_strikes)+1])
    
# computed binomial american option trees for every simulation (strike and stock starting value)
    
call_8_m = np.zeros([NSim, N+1, N+1])   

for k in range(int(NSim)):  
    call_8_m[k,:, int(N-enter_prices[k,1])] = np.maximum(S_bin_m8[k,:,int(N-enter_prices[k,1])]-strikes[k],0)
    
for k in range(int(NSim)):
    for i in np.arange(int(N-enter_prices[k,1]-1), -1, -1):
        for j in np.arange(0, i+1):
            call_8_m[k,j,i] = np.maximum(np.maximum(S_bin_m8[k,j,i]-strikes[k],0),
                                         exp(-r*dt)*(p*call_8_m[k,j,i+1]+(1-p)*call_8_m[k,j+1,i+1]))

# computes ESO fair value at t=0

payoff8 = call_8_m[:,0,0]*np.exp(-r*enter_prices[:,1]*dt)
call8 = mean(payoff8)

print("Task 8) Point estimate:", call8)
print ("Task 8) MC radius:" , np.std(payoff8)*1.96/(sqrt(NSim)))
print("Task 8) 95% CI:" , [ call8 - np.std(payoff8)*1.96/(sqrt(NSim)), 
                   call8 + np.std(payoff8)*1.96/(sqrt(NSim))])
print ("Task 8) MC radius/point estimate:" , np.std(payoff8)*1.96/(sqrt(NSim))/call8) 

