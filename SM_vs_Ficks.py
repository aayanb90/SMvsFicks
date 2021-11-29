# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:12:31 2021

@author: Dr.-Ing. A. Banerjee (a.banerjee@utwente.nl)

This code is to be used for educational purposes only! 
Please do not use/distribute without prior permission from the author.

"""

import numpy as np
import time as t
import matplotlib.pyplot as plt
import math as m
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#Gas properties
Vh2 = 6.12
Vh2o = 13.1
Vo2 = 16.3
Vn2 = 18.5
Vco2 = 26.7
Mh2 = 1.00794*2*1e-3
Mh2o = 18.01528*1e-3
Mo2 = 31.999*1e-3
Mn2 = 14.0067*2*1e-3
Mco2 = 44.01*1e-3
Mh2h2o = 2/(1/Mh2+1/Mh2o)
Mh2o2 = 2/(1/Mh2+1/Mo2)
Mo2h2o = 2/(1/Mo2+1/Mh2o)
Mo2n2 = 2/(1/Mo2+1/Mn2)
Mh2n2 = 2/(1/Mh2+1/Mn2)
Mh2co2 = 2/(1/Mh2+1/Mco2)

patm = 1.01325e5
pin = patm
T = 298.15
R = 8.314
ctotin = pin/(R*T)
u = 1e-2 #m/s

Dh2h2o = ((1e-4*0.00143*T**1.75)/((Mh2h2o*1e3)**(1/2)*(Vh2**(1/3)+Vh2o**(1/3))**2))/(pin/1e5)
Dh2o2 = ((1e-4*0.00143*T**1.75)/((Mh2o2*1e3)**(1/2)*(Vh2**(1/3)+Vo2**(1/3))**2))/(pin/1e5)
Do2h2o = ((1e-4*0.00143*T**1.75)/((Mo2h2o*1e3)**(1/2)*(Vo2**(1/3)+Vh2o**(1/3))**2))/(pin/1e5)
Do2n2 = ((1e-4*0.00143*T**1.75)/((Mo2n2*1e3)**(1/2)*(Vo2**(1/3)+Vn2**(1/3))**2))/(pin/1e5)
Dh2n2 = ((1e-4*0.00143*T**1.75)/((Mh2n2*1e3)**(1/2)*(Vh2**(1/3)+Vn2**(1/3))**2))/(pin/1e5)
Dh2co2 = ((1e-4*0.00143*T**1.75)/((Mh2co2*1e3)**(1/2)*(Vh2**(1/3)+Vco2**(1/3))**2))/(pin/1e5)

L = 1e-2 #m
nx = 25
dx = L/nx
k_reac = 1e-1 #(low H2O yield)
# k_reac = 1 #(mid H2O yield)
# k_reac = 3 #(high H2O yield)
K_eq = 100
Xh2oin = 0
Xo2in = 0.4
Xh2in = 1- Xo2in - Xh2oin
Meffin = Xh2oin*Mh2o + Xo2in*Mo2 + Xh2in*Mh2

rh2opos = np.arange(0,1*nx)
ro2pos = np.arange(nx,2*nx)
rtotpos = np.arange(2*nx,3*nx)

svrini = np.zeros(3*nx)
svrini[rtotpos] = ctotin*Mh2

def fc_rtrans_Ficks(svr):
    rres_t_Ficks = np.zeros(3*nx)
    rh2o = svr[rh2opos]
    ro2 = svr[ro2pos]
    rtot = svr[rtotpos]
    Yh2o = rh2o/rtot
    Yo2 = ro2/rtot
    Yh2 = 1 - Yh2o - Yo2
    moltot = Yh2o/Mh2o + Yo2/Mo2 + Yh2/Mh2
    Xh2o = (Yh2o/Mh2o)/moltot
    Xo2 = (Yo2/Mo2)/moltot
    Xh2 = (Yh2/Mh2)/moltot
    Meff = Xh2o*Mh2o + Xo2*Mo2 + Xh2*Mh2
    ctot = rtot/Meff
    reac = k_reac*Xh2*Xo2**0.5 - (k_reac/K_eq)*Xh2o
    sh2o = reac*ctot*Mh2o
    so2 = -0.5*reac*ctot*Mo2
    sh2 = -reac*ctot*Mh2
    stot = sh2o + so2 + sh2
    Xh2o = np.insert(Xh2o,0,Xh2oin)
    ctot = np.insert(ctot,0,ctotin)
    Jxh2o = -Dh2h2o*np.diff(ctot*Xh2o)/dx
    Jxh2o[0] = Jxh2o[0]/2
    Jxh2o = np.insert(Jxh2o,len(Jxh2o),0)
    Jxh2o = Jxh2o + u*Xh2o*ctot
    Xo2 = np.insert(Xo2,0,Xo2in)
    Jxo2 = -Dh2o2*np.diff(ctot*Xo2)/dx
    Jxo2[0] = Jxo2[0]/2
    Jxo2 = np.insert(Jxo2,len(Jxo2),0)
    Jxo2 = Jxo2 + u*Xo2*ctot
    rres_t_Ficks[rh2opos] = (-np.diff(Jxh2o*Mh2o)/dx) + sh2o
    rres_t_Ficks[ro2pos] = (-np.diff(Jxo2*Mo2)/dx) + so2
    rtot = np.insert(rtot,0,ctotin*Meffin)
    rres_t_Ficks[rtotpos] = -np.diff(u*rtot)/dx + stot
    return rres_t_Ficks

start = t.time()
svrsol_f = fsolve(fc_rtrans_Ficks,svrini)
rgt_f = svrsol_f
Yh2ot_f = rgt_f[rh2opos]/rgt_f[rtotpos]
Yo2t_f = rgt_f[ro2pos]/rgt_f[rtotpos]
Yh2t_f = 1 - Yh2ot_f - Yo2t_f
moltot_f = Yh2ot_f/Mh2o + Yo2t_f/Mo2 + Yh2t_f/Mh2
Xh2ot_f = (Yh2ot_f/Mh2o)/moltot_f
Xo2t_f = (Yo2t_f/Mo2)/moltot_f
Xh2t_f = (Yh2t_f/Mh2)/moltot_f
Xh2ot_f = np.insert(Xh2ot_f,0,Xh2oin,axis=0)
Xo2t_f = np.insert(Xo2t_f,0,Xo2in,axis=0)
Xh2t_f = np.insert(Xh2t_f,0,Xh2in,axis=0)
Xtot_f = Xh2ot_f + Xo2t_f + Xh2t_f

def fc_D_SM(Xh2o,Xo2):
    H = np.zeros((2,2))
    H[0,0] = 1/Dh2h2o + ((1/Do2h2o)-(1/Dh2h2o))*Xo2
    H[0,1] = -((1/Do2h2o)-(1/Dh2h2o))*Xh2o
    H[1,0] = -((1/Do2h2o)-(1/Dh2o2))*Xo2
    H[1,1] = 1/Dh2o2 + ((1/Do2h2o)-(1/Dh2o2))*Xh2o
    D_SM = np.linalg.inv(H)
    return D_SM

def fc_rtrans_SM(svr):
    rres_t_SM = np.zeros(3*nx)
    rh2o = svr[rh2opos]
    ro2 = svr[ro2pos]
    rtot = svr[rtotpos]
    Yh2o = rh2o/rtot
    Yo2 = ro2/rtot
    Yh2 = 1 - Yh2o - Yo2
    moltot = Yh2o/Mh2o + Yo2/Mo2 + Yh2/Mh2
    Xh2o = (Yh2o/Mh2o)/moltot
    Xo2 = (Yo2/Mo2)/moltot
    Xh2 = (Yh2/Mh2)/moltot
    Meff = Xh2o*Mh2o + Xo2*Mo2 + Xh2*Mh2
    ctot = rtot/Meff
    reac = k_reac*Xh2*Xo2**0.5 - (k_reac/K_eq)*Xh2o
    sh2o = reac*ctot*Mh2o
    so2 = -0.5*reac*ctot*Mo2
    sh2 = -reac*ctot*Mh2
    stot = sh2o + so2 + sh2
    Jxh2o = np.zeros(nx+1)
    Jxo2 = np.zeros(nx+1)
    Xh2o = np.insert(Xh2o,0,Xh2oin)
    Xo2 = np.insert(Xo2,0,Xo2in)
    gradXh2o = np.diff(Xh2o)/dx
    gradXo2 = np.diff(Xo2)/dx
    for i in range(nx):
        Diff_SM = fc_D_SM(Xh2o[i],Xo2[i])
        Jxh2o[i] = -Diff_SM[0,0]*gradXh2o[i]-Diff_SM[0,1]*gradXo2[i]
        Jxo2[i] = -Diff_SM[1,0]*gradXh2o[i]-Diff_SM[1,1]*gradXo2[i]
    Jxh2o[0] = Jxh2o[0]/2
    Jxh2o = Jxh2o + u*Xh2o
    Jxo2[0] = Jxo2[0]/2
    Jxo2 = Jxo2 + u*Xo2
    rres_t_SM[rh2opos] = ((-np.diff(Jxh2o)/dx)*ctot*Mh2o) + sh2o
    rres_t_SM[ro2pos] = ((-np.diff(Jxo2)/dx)*ctot*Mo2) + so2
    rtot = np.insert(rtot,0,ctotin*Meffin)
    rres_t_SM[rtotpos] = -np.diff(u*rtot)/dx + stot
    return rres_t_SM

start = t.time()

svrsol_sm = fsolve(fc_rtrans_SM,svrini)
rgt_sm = svrsol_sm
Yh2ot_sm = rgt_sm[rh2opos]/rgt_sm[rtotpos]
Yo2t_sm = rgt_sm[ro2pos]/rgt_sm[rtotpos]

Yh2t_sm = 1 - Yh2ot_sm - Yo2t_sm
moltot_sm = Yh2ot_sm/Mh2o + Yo2t_sm/Mo2 + Yh2t_sm/Mh2
Xh2ot_sm = (Yh2ot_sm/Mh2o)/moltot_sm
Xo2t_sm = (Yo2t_sm/Mo2)/moltot_sm
Xh2t_sm = (Yh2t_sm/Mh2)/moltot_sm
Xh2ot_sm = np.insert(Xh2ot_sm,0,Xh2oin,axis=0)
Xo2t_sm = np.insert(Xo2t_sm,0,Xo2in,axis=0)
Xh2t_sm = np.insert(Xh2t_sm,0,Xh2in,axis=0)
Xtot_sm = Xh2ot_sm + Xo2t_sm + Xh2t_sm

D_SM_o = fc_D_SM(Xh2ot_sm[-1],Xo2t_sm[-1])
D_error_h2o = Dh2h2o/D_SM_o[0,0]
D_error_o2 = Dh2o2/D_SM_o[1,1]


# %% Plot Results

xgrid = np.arange(0,L*1e2+(L*1e2/nx),(L*1e2/nx))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Steady-state H2O Mole fraction along channel',pad=15,x=0.54)
ax.set_xlim(0,L*1e2)
ax.set_xlabel('Channel Length (cm)')
ax.set_ylabel('Mole Fraction (-)')
ax.set_ylim(0,1.1*np.amax(Xh2ot_sm))
line1, = ax.plot(xgrid,Xh2ot_f)
line2, = ax.plot(xgrid,Xh2ot_sm)
labels = ['Ficks','Stefan-Maxwell']
ax.legend([line1,line2],labels,loc="lower right")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Steady-state O2 Mole fractions along channel',pad=15,x=0.54)
ax.set_xlim(0,L*1e2)
ax.set_xlabel('Channel Length (cm)')
ax.set_ylabel('Mole Fraction (-)')
ax.set_ylim(0,1.1*np.amax(Xo2t_f))
line1, = ax.plot(xgrid,Xo2t_f)
line2, = ax.plot(xgrid,Xo2t_sm)
labels = ['Ficks','Stefan-Maxwell']
ax.legend([line1,line2],labels,loc="lower right")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Steady-state H2 Mole fractions along channel',pad=15,x=0.54)
ax.set_xlim(0,L*1e2)
ax.set_xlabel('Channel Length (cm)')
ax.set_ylabel('Mole Fraction (-)')
ax.set_ylim(0,1.1*np.amax(Xh2t_f))
line1, = ax.plot(xgrid,Xh2t_f)
line2, = ax.plot(xgrid,Xh2t_sm)
labels = ['Ficks','Stefan-Maxwell']
ax.legend([line1,line2],labels,loc="lower right")

# # %% Plot Errors

# xgrid = np.arange(0,L*1e2+(L*1e2/nx),(L*1e2/nx))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Steady-state Error in H2O Mole fraction along channel',pad=15,x=0.54)
# ax.set_xlim(0,L*1e2)
# ax.set_xlabel('Channel Length (cm)')
# ax.set_ylabel('Relative Error (%)')
# ax.plot(xgrid,(abs(Xh2ot_f-Xh2ot_sm)/Xh2ot_sm)*1e2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Steady-state Error in O2 Mole fractions along channel',pad=15,x=0.54)
# ax.set_xlim(0,L*1e2)
# ax.set_xlabel('Channel Length (cm)')
# ax.set_ylabel('Relative Error (%)')
# ax.plot(xgrid,(abs(Xo2t_f-Xo2t_sm)/Xo2t_sm)*1e2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Steady-state Error in H2 Mole fractions along channel',pad=15,x=0.54)
# ax.set_xlim(0,L*1e2)
# ax.set_xlabel('Channel Length (cm)')
# ax.set_ylabel('Relative Error (%)')
# ax.plot(xgrid,(abs(Xh2t_f-Xh2t_sm)/Xh2t_sm)*1e2)