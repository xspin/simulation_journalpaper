#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
import sim

per_a = [274.7229, 90.2514, 67.6181, 53.3987, 35.3508]
per_g = [7.9932, 3.4998, 1.6883, 0.3756, 0.0900]
per_n = 2
per_an, per_gn = per_a[per_n], per_g[per_n]

def loss(qi, qj):
    parameter = sim.parameter
    rho = parameter['rho']
    t = np.sum(np.square(qi-qj))
    return rho/t

def PER(sinr):
    # pn = np.exp(an)/gn 
    an, gn = per_an, per_gn
    return min(an*np.exp(-gn*sinr), 1) #if sinr>pn else 1

def gain(var, i, j):
    parameter = sim.parameter
    assert(i!=j)
    Q = var['position']
    rho = parameter['rho']
    d2 = np.sum(np.square(Q[i]-Q[j]))
    return rho/d2


def floyd(mdelay):
    N = len(mdelay)
    path = np.ones([N,N], dtype=int) * -1
    for k in range(N):
        for i in range(N):
            if i==k: continue
            for j in range(N):
                if i==j or j==k: continue
                t = mdelay[i,k] + mdelay[k,j]
                if t==np.inf: continue
                if np.abs(t-mdelay[i,j])<1e-5:
                    pass
                elif t < mdelay[i,j]:
                    mdelay[i,j] = t
                    path[i,j] = k
    return mdelay, path


def SINR_PER_matrix(var):
    parameter = sim.parameter
    Q, P = var['position'], var['power']
    N = len(P)
    sinr = np.zeros([N, N]) 
    per = np.ones([N, N]) 
    interf = np.zeros(N)
    for i in range(N):
        for j in range(i+1, N):
            hij = loss(Q[i], Q[j])
            interf[i] += P[j]*hij
            interf[j] += P[i]*hij
    r = parameter['probability']
    interf *= r
    interf += parameter['noise']
    for i in range(N):
        for j in range(N):
            if i==j: 
                sinr[i,j] = 10 
                per[i,j] = 0
            else:
                hij = loss(Q[i], Q[j])
                sinr[i, j] = 10*np.log10(P[i]*hij/(interf[j]-r*P[i]*hij)) #dB
                per[i, j] = PER(sinr[i,j])
    return sinr, per

def check(per, mper, path):
    iserror = False
    print('Checking....')
    for i in range(len(path)):
        for j in range(len(path)):
            k = path[i][j]
            if k==i or k==j:
                print('Error in {}=>{}'.format(i,j))
                iserror = True
            if mper[i, j]-per[i,j]>1e-5:
                print('Error in {}=>{}'.format(i,j))
                print('  mper:{}, per:{}'.format(mper[i,j], per[i,j]))
                iserror = True
    print('Check over.')
    if iserror:
        exit('error!')
        
def delay_graph(var):
    parameter = sim.parameter
    T = parameter['RTT']
    dm = parameter['delay_h2m']
    dh = parameter['delay_l2h']
    d = max(dm, dh)
    sinr, per = SINR_PER_matrix(var)
    N = len(per)
    # delay = T/(1-per)
    # delay[delay>0] = np.inf
    delay = np.array([[T/(1-per[i,j]) if per[i,j]<0.9999 else np.inf for j in range(N)] for i in range(N)])
    for i in range(N): delay[i,i] = 0
    mdelay = np.copy(delay)
    mdelay, path = floyd(mdelay)
    return sinr, per, delay, mdelay, path

def graph(var, delay, mdelay, path):
    parameter = sim.parameter
    debug = False
    T = parameter['RTT']
    dm = parameter['delay_h2m']
    dh = parameter['delay_l2h']
    D = max(dm, dh)
    H = var['head']
    N = len(H)
    ch = np.array([i for i in range(N) if H[i]])
    # path from CHs to CMs
    mpath = np.ones(N, dtype='int')*-1
    mpath[ch] = -2
    head = np.ones(N, dtype='int')*-1
    for i in range(N):
        if H[i]: continue
        j = np.argmin(mdelay[ch, i])
        j = ch[j]
        if mdelay[j,i] == np.inf: continue
        head[i] = j
        if (mdelay[j,i]>1000):
            print('Warnning: mdelay', j, i, mdelay[j,i])
            # exit()
        cnt = N
        if debug: print('{:3d} =>{:3d}:'.format(j, i), j, '->', end=' ')
        while path[j, i]>0 and cnt>=0:
            j = path[j, i]
            if debug: print(j, '->', end=' ')
            cnt -= 1
        mpath[i] = j 
        if debug: print(i)
        if path[j,i]>0:
            print('Error!')
            exit()
    # path from L to CHs
    hpath = np.ones(N, dtype='int')*-1
    # idx = np.argsort(per[0,:])
    for i in range(N):
        if i==0: continue
        j = 0
        while path[j, i]>=0: j = path[j,i]
        if delay[j,i] < np.inf:
            hpath[i] = j
    temp = np.ones(N, dtype=int) * -1
    for i in ch:
        j = i
        while j>0: 
            temp[j] = hpath[j]
            j = hpath[j]
    hpath = temp
    return mpath, hpath, head

def plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=True, save=False, filename=None, title=None):
    parameter = sim.parameter
    plot_id = False
    plot_per = False
    plot_proj = False
    plot_delay = True
    plot_velocity = False
    plot_power = True
    # print('\nploting...')
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    N = len(P)
    plt.figure()
    plt.title(title)
    width = parameter['width']
    R = parameter['RC']
    xy_max = np.max(Q, 0)
    xy_min = np.min(Q, 0)
    # plt.xlim(-width, width)
    # plt.ylim(-width, width)
    # plt.xlim(xy_min[0]-5, xy_max[0]+5)
    # plt.ylim(xy_min[1]-5, xy_max[1]+5)
    xlim = [min(-width, xy_min[0]-R), max(width, xy_max[0]+R)]
    ylim = [min(-width, xy_min[1]-R), max(width, xy_max[1]+R)]
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.plot(*Q[0], marker='*', ls='', mfc='r', ms=13)
    if plot_velocity:
        plt.arrow(*Q[0], *(V*3), head_width=5, head_length=6, fc='k', ec='k')
        plt.text(*(V*3), '({:.2f}, {:.2f})'.format(*V), color='r')
    
    # plot coverage circles
    t = np.linspace(0, 2*np.pi, 20)
    txy = R*np.array([np.cos(t), np.sin(t)])
    for i in range(N):
        plt.plot(Q[i,0]+txy[0], Q[i,1]+txy[1], 'g--')

    # plot power circles 
    if plot_power:
        t = np.linspace(0, 2*np.pi, 20)
        temp = np.array([np.cos(t), np.sin(t)])
        for i in range(N):
            txy = P[i]*temp/2
            plt.plot(Q[i,0]+txy[0], Q[i,1]+txy[1], 'b--')
            # plt.text(Q[i,0]-10, Q[i,1]+10, '{:.1f}'.format(P[i]), color='b')

    delaystr = lambda r: '{:.2f}'.format(r) if r>=1.005 else '1'
    # print(hpath)

    # plot nodes
    for i in range(1, N):
        #plot ID
        if plot_id: plt.text(*(Q[i]+[1,1]), '{}'.format(i), color='b')

        j = mpath[i]
        k = hpath[i]
        if j==-1: # orphan nodes
            plt.plot(*Q[i], marker='o', mfc='b', ms=7)
        elif j==-2: # cluster heads
            if k>=0:
                # CHs connected to the leader
                plt.plot(*Q[i], marker='s', mfc='r', ms=7)
            else:
                plt.plot(*Q[i], marker='s', mfc='b', ms=7)
        else: # cluster members
            # CMs connected to the CHs
            plt.plot(*Q[i], marker='o', mfc='k', ms=7)
        if H[i]: 
            # CH i <- pre k
            if k>=0: # show the delay to the leader
                plt.text(*Q[i]-[0,3], delaystr(mdelay[0][i]), color='r')
        elif j>=0:
            # i <- j
            # x = np.r_[Q[i, 0], Q[j, 0]]
            # y = np.r_[Q[i, 1], Q[j, 1]]
            # plt.plot(x,y, marker='', ls='-', c='g')
            # plt.text(*Q[i]-[3,0], delaystr(mdelay[head[i]][i]), color='g')
            plt.arrow(*Q[j], *((Q[i]-Q[j])/5), head_width=5, head_length=6, fc='purple', ec='purple')
        if k>=0:
            # arrow from k to i (CL to CH)
            plt.arrow(*Q[k], *((Q[i]-Q[k])/5), head_width=5, head_length=6, fc='r', ec='r')
            if k>0: plt.text(*Q[k]-[0,3], delaystr(mdelay[0][k]), color='r')
        if head[i]>=0:
            # delay from CH to CM i
            plt.text(*Q[i]-[3,0], delaystr(mdelay[head[i]][i]), color='purple')

    if plot_delay:
        D = 10
        for i in range(N):
            for j in range(i+1, N):
                if delay[i,j]<D or delay[j,i]<D:
                    x = np.r_[Q[i, 0], Q[j, 0]]
                    y = np.r_[Q[i, 1], Q[j, 1]]
                    plt.plot(x,y, marker='', ls=':', c='grey')
                    z = (Q[i]+Q[j])/2
                    d = norm(Q[i]-Q[j])
                    # plt.text(z[0], z[1], '{:.1f}m'.format(d), color='grey')

                if delay[i,j]<D: # i->j
                    z = Q[j] + (Q[i]-Q[j])/3
                    # text = '({:.1f}dB, {:.1f})'.format(sinr[i,j], per[i,j])
                        # text = '{:.3f}'.format(per[i,j])
                    text = delaystr(delay[i,j])
                    plt.text(*z, text, color='grey')
                    # plt.arrow(*Q[i], *((Q[j]-Q[i])/10), head_width=0.6, head_length=0.7, fc='g', ec='grey')
                if delay[j,i]<D: # j->i
                    z = Q[i] + (Q[j]-Q[i])/3
                    # text = '({:.1f}dB, {:.1f})'.format(sinr[j,i], per[j,i])
                        # text = '{:.3f}'.format(per[j,i])
                    text = delaystr(delay[j,i])
                    plt.text(z[0], z[1], text, color='grey')
                    # plt.arrow(*Q[j], *((Q[i]-Q[j])/5), head_width=0.6, head_length=0.7, fc='g', ec='grey')
    if plot_per:
        for i in range(N):
            for j in range(i+1, N):
                if per[i,j]<0.99 or per[j,i]<0.99:
                    x = np.r_[Q[i, 0], Q[j, 0]]
                    y = np.r_[Q[i, 1], Q[j, 1]]
                    plt.plot(x,y, marker='', ls='--', c='grey')
                    z = (Q[i]+Q[j])/2
                    d = norm(Q[i]-Q[j])
                    # plt.text(z[0], z[1], '{:.1f}m'.format(d), color='grey')

                if per[i,j]<0.999: # i->j
                    z = Q[j] + (Q[i]-Q[j])/3
                    # text = '({:.1f}dB, {:.1f})'.format(sinr[i,j], per[i,j])
                        # text = '{:.3f}'.format(per[i,j])
                    text = delaystr(per[i,j])
                    plt.text(*z, text, color='grey')
                    # plt.arrow(*Q[i], *((Q[j]-Q[i])/10), head_width=0.6, head_length=0.7, fc='g', ec='grey')
                if per[j,i]<0.999: # j->i
                    z = Q[i] + (Q[j]-Q[i])/3
                    # text = '({:.1f}dB, {:.1f})'.format(sinr[j,i], per[j,i])
                        # text = '{:.3f}'.format(per[j,i])
                    text = delaystr(per[j,i])
                    plt.text(z[0], z[1], text, color='grey')
                    # plt.arrow(*Q[j], *((Q[i]-Q[j])/5), head_width=0.6, head_length=0.7, fc='g', ec='grey')

    # plot the projection
    if plot_proj:
        v_vert = np.r_[V[1], -V[0]]/norm(V)
        u = np.dot(Q, v_vert)
        # u = np.sort(u)
        ps = np.array([x*v_vert for x in u])
        plt.plot(ps[:,0], ps[:,1], '--', c='grey', marker='o', ms=3, mfc='r')
        # plot v_vert
        plt.arrow(*Q[0], *(v_vert*7), head_width=1, head_length=1, fc='r', ec='r')
        # plt.text(*(v_vert*3), '({:.2f}, {:.2f})'.format(*v_vert), color='r')

    # plt.title('{} nodes'.format(N))
    # t = time.localtime()
    # filename = '{}{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday)
    if save:
        t = time.time()
        if not filename:
            filename = "img/{:.0f}-{}nodes.png".format(t, N)
        plt.savefig(filename)
    if show:
        plt.show()
        print('closed.\n')
    plt.close()
