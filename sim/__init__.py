#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time, datetime

def mw2dbm(p):
    # p mW
    return 10*np.log10(p) # dBm

def dbm2mw(a):
    # a dBm
    return 10**(a/10.0) #mW

figpath = '.'

parameter = {
    'width': 100,
    '#nodes': 23,
    'RC': 50.0,
    'rho': 1.0,     #ratio
    'noise': -30,  #dBm
    'P': 0,   #mW
    'Pmax': 100.0,   #mW
    'Pmin': 20.0,   #mW
    'Vmax': 10.0,   #m/s
    'weight': 5.0,  #kg
    'c_resis': 1.0,
    'c_force': 1.0,
    'probability': 0.05,
    'delay_h2m': 4.0,
    'delay_l2h': 6.0,
    'lambda1': 100.0,
    'lambda2': 100.0,
    'lambda3': 100.0,
    'm': 1.0,
    'g': 9.8,
    'RTT': 1.0,
    'a': 0.9
}
parameter['noise'] = dbm2mw(parameter['noise']) # convert to mw

# start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
sim_start_time = time.time()

# def sim.param_set(key, val):

def sec2date(sec):
    sec = int(sec)
    return '{}:{}:{}'.format(sec//3600, sec//60%60, sec%60)

def difftime():
    # end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    # cost_time = (datetime.datetime.strptime(end_time,'%H:%M:%S') - datetime.datetime.strptime(start_time,'%H:%M:%S'))
    cost = time.time() - sim_start_time
    return sec2date(cost)

def initialize(type='random', heads=False):
    '''
    input: 
    output: Q, P, H, V
    '''
    N = parameter['#nodes']
    R = parameter['RC']
    # width = np.sqrt(N)*50
    width = parameter['width']
    Q = np.random.random([N,2])
    # initialize position
    if type is 'random':
        Q = np.random.random([N, 2]) #-w/2 ~ w/2
        Q = Q*width - width/2.0
    elif type is 'line':
        Q = np.c_[np.random.random([N, 1])*width-width/2.0, [0]*N]
    elif type is 'circle':
        pass
    elif type is 'grid':
        n = np.round(np.sqrt(N)).astype(int)
        m = np.ceil(N/n).astype(int)
        # if m*n<N: m+=1
        print(n,m)
        for i in range(n):
            for j in range(m):
                k = i*m+j
                if k>=N: break
                Q[k] = np.r_[(i-n//2), (j-m//2)]*R*2
                if i==n//2 and j==m//2:
                    Q[k] = Q[0]
    elif type is 'greedy':
        pass
    elif type is 'dot':
        Q = np.random.random([N, 2])
        Q = Q*10 - 5
    # for k in range(1, np.ceil(np.log2(N+1))):

    Q[0] = [0,0]
    # print(Q)

    #exit()
    # initialize power

    if parameter['P']<=1:
        # P = np.ones(N)*
        mu = parameter['Pmax']*parameter['P']
    else:
        # P = np.ones(N)*parameter['P']
        mu = parameter['P']
    P = np.random.normal(mu, 10, N)
    if parameter['P']<=0:
        P = np.random.rand(N)*80+20

    # H = np.zeros(N)
    if heads:
        H = np.array([True if np.random.random()<0.2 else False for i in range(N)])
    else:
        H = np.array([False for i in range(N)])
    H[0] = True 
    theta = np.random.random()*2*np.pi 
    theta = np.pi/4
    V = np.r_[np.cos(theta), np.sin(theta)]* parameter['Vmax']/2
    # return Q, P, H, V
    var = {}
    var['position'], var['power'], var['head'], var['velocity'] = Q, P, H, V
    return var
